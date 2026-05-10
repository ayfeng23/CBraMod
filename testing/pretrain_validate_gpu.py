"""Quick GPU validation for CBraMod pretraining.

Loads the merged TUEG LMDB, runs the same masked-reconstruction inner loop as
pretrain_trainer.Trainer for N iterations, and reports throughput, GPU memory,
and loss trajectory. Validates dataloader, model, autograd, optimizer step on
a real GPU before committing to a long run.
"""

import argparse
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/ayf4/cbramod')
from datasets.pretraining_dataset import PretrainingDataset  # noqa: E402
from models.cbramod import CBraMod                           # noqa: E402
from utils.util import generate_mask                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/home/ayf4/scratch_pi_zf59/ayf4/data/tueg_v2.0.1_lmdb')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--mask_ratio', type=float, default=0.5)
    ap.add_argument('--clip', type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f'device: {device}  cuda_available: {torch.cuda.is_available()}')
    if device.type == 'cuda':
        print(f'gpu:   {torch.cuda.get_device_name(device)}')
        print(f'mem:   {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB total')

    ds = PretrainingDataset(args.data)
    print(f'dataset: {len(ds):,} samples')
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    shuffle=True, pin_memory=(device.type == 'cuda'))

    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                    seq_len=30, n_layer=12, nhead=8).to(device)
    print(f'model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2)
    crit = nn.MSELoss(reduction='mean')

    losses = []
    t0 = time.time()
    last = t0
    for i, x in enumerate(dl):
        if i >= args.iters:
            break
        t_data = time.time() - last

        opt.zero_grad()
        x = x.to(device, non_blocking=True) / 100
        bz, ch, n_patch, _ = x.shape
        mask = generate_mask(bz, ch, n_patch, mask_ratio=args.mask_ratio, device=device)
        y = model(x, mask=mask)
        loss = crit(y[mask == 1], x[mask == 1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t_iter = time.time() - last
        last = time.time()

        losses.append(loss.item())
        if i < 5 or (i + 1) % 10 == 0:
            print(f'iter {i:4d}: loss={loss.item():7.3f}  '
                  f't_iter={t_iter * 1000:5.0f}ms  t_data={t_data * 1000:5.0f}ms')

    total = time.time() - t0
    n = len(losses)
    print(f'\n=== {n} iters in {total:.1f}s ===')
    print(f'throughput: {n * args.batch_size / total:.0f} samples/s  ({n / total:.2f} iter/s)')
    print(f'loss: first={losses[0]:.3f}  last={losses[-1]:.3f}  '
          f'min={min(losses):.3f}  delta={losses[0] - losses[-1]:+.3f}')
    if device.type == 'cuda':
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f'gpu peak mem: {peak_gb:.2f} GB')

    iters_per_epoch = len(ds) // args.batch_size
    epoch_min = iters_per_epoch / (n / total) / 60
    print(f'\nepoch estimate: {iters_per_epoch:,} iters / epoch  ~{epoch_min:.1f} min/epoch  '
          f'~{40 * epoch_min / 60:.1f} hr / 40 epochs')


if __name__ == '__main__':
    main()
