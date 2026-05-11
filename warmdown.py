import math
import os
import re
import time

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.pretraining_dataset import PretrainingDataset
from models.cbramod import CBraMod
from utils.util import generate_mask


PRETRAIN_ARCH = dict(
    in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
    seq_len=30, n_layer=12, nhead=8,
)


def _read_current_lr_from_full_ckpt(full_ckpt_path):
    full = torch.load(full_ckpt_path, map_location='cpu', weights_only=False)
    return full['optimizer']['param_groups'][0]['lr']


def _cosine_lr_at_epoch_boundary(epoch, total_epochs=40, base_lr=5e-4, eta_min=1e-5):
    return eta_min + (base_lr - eta_min) * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))


def _parse_epoch_from_filename(ckpt_path):
    name = os.path.basename(ckpt_path)
    m = re.match(r'epoch(\d+)_loss', name)
    return int(m.group(1)) if m else None


def warmdown(ckpt_path, output_dir, tueg_data_dir, *,
             batch_size=128, num_workers=8, clip_value=1.0, mask_ratio=0.5,
             cuda=0, weight_decay=5e-2, objective='recon', need_mask=True):
    """Run 1 epoch on TUEG with LR linearly decaying current_lr -> 0.
    `objective`/`need_mask` must match the pretraining run that produced ckpt_path
    (mirrors pretrain_trainer.py:111-138; ntp forces need_mask=False)."""
    if objective not in ('recon', 'ntp'):
        raise ValueError(f"objective must be 'recon' or 'ntp', got {objective!r}")
    if objective == 'ntp':
        need_mask = False 

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    full_ckpt_path = ckpt_path + '.full.pth'
    if os.path.exists(full_ckpt_path):
        starting_lr = _read_current_lr_from_full_ckpt(full_ckpt_path)
        lr_source = full_ckpt_path
    else:
        epoch_in_name = _parse_epoch_from_filename(ckpt_path)
        if epoch_in_name is not None:
            starting_lr = _cosine_lr_at_epoch_boundary(epoch_in_name)
            lr_source = f'cosine analytical @ epoch={epoch_in_name}'
        else:
            raise ValueError(
                f"Cannot determine warm-down starting LR: no '{full_ckpt_path}' "
                f"companion and could not parse 'epochN_loss' from '{ckpt_path}'. "
                "Save the .full.pth alongside the checkpoint or rename the file."
            )
    print(f'[warmdown] starting LR {starting_lr:.6e} (from {lr_source})')

    ds = PretrainingDataset(dataset_dir=tueg_data_dir)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    data_length = len(loader)
    print(f'[warmdown] TUEG dataset {len(ds)} samples, {data_length} batches @ bs={batch_size}')

    model = CBraMod(**PRETRAIN_ARCH, objective=objective).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    print(f'[warmdown] objective={objective}  need_mask={need_mask}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=starting_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=data_length,
    )

    criterion = MSELoss(reduction='mean').to(device)
    model.train()
    losses = []
    t0 = time.time()
    for x in tqdm(loader, mininterval=10, desc='warmdown'):
        optimizer.zero_grad()
        x = x.to(device) / 100
        if need_mask:
            bz, ch_num, patch_num, _ = x.shape
            mask = generate_mask(bz, ch_num, patch_num, mask_ratio=mask_ratio, device=device)
            y = model(x, mask=mask)
            masked_x = x[mask == 1]
            masked_y = y[mask == 1]
            loss = criterion(masked_y, masked_x)
        elif objective == 'recon':
            y = model(x)
            loss = criterion(y, x)
        else:  # objective == 'ntp'
            y = model(x[:, :, :-1, :])
            labels = x[:, :, 1:, :]
            loss = criterion(y, labels)
        loss.backward()
        if clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
    wall = time.time() - t0
    mean_loss = sum(losses) / max(len(losses), 1)
    final_lr = optimizer.param_groups[0]['lr']
    print(f'[warmdown] mean_loss {mean_loss:.6f}, final LR {final_lr:.2e}, wall {wall/60:.1f} min')

    out_path = os.path.join(output_dir, 'warmdown.pth')
    torch.save(model.state_dict(), out_path)
    print(f'[warmdown] wrote {out_path}')
    try:
        ds.db.close()
    except Exception:
        pass
    return out_path, {
        'mean_loss': mean_loss,
        'starting_lr': float(starting_lr),
        'final_lr': float(final_lr),
        'wall_seconds': wall,
        'data_length': data_length,
    }
