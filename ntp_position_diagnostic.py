"""1-epoch NTP diagnostic: track per-time-position MSE over the course of training,
plus the trivial 'copy previous patch' baseline. Helps answer whether NTP loss
saturation is uniform across positions or concentrated at the easier (later)
positions where context is rich.
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.pretraining_dataset import PretrainingDataset
from models.cbramod import CBraMod


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', default='/home/ayf4/scratch_pi_zf59/ayf4/data/tueg_v2.0.1_lmdb')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--cuda', type=int, default=0)
    p.add_argument('--log-window', type=int, default=200,
                   help='Window size in batches over which per-position MSE is averaged before logging.')
    p.add_argument('--clip-value', type=float, default=1.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.cuda)

    ds = PretrainingDataset(dataset_dir=args.dataset_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    n_batches = len(loader)
    print(f'[diag] TUEG: {len(ds)} samples, {n_batches} batches @ bs={args.batch_size}')

    model = CBraMod(
        in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
        seq_len=30, n_layer=12, nhead=8, objective='ntp',
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2)

    history = []
    window_per_pos = None
    window_avg = 0.0
    window_count = 0
    baseline_per_pos = None
    last_batch_idx = -1
    out_path = os.path.join(args.output_dir, 'results.json')

    def _dump(final=False):
        with open(out_path, 'w') as f:
            json.dump({
                'baseline_per_position_mse': baseline_per_pos,
                'history': history,
                'wall_seconds': time.time() - t0,
                'wall_minutes': (time.time() - t0) / 60.0,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'num_batches': n_batches,
                'seq_len': 30,
                'num_target_positions': 29,
                'log_window': args.log_window,
                'last_batch_idx': last_batch_idx,
                'finished': final,
            }, f, indent=2)

    model.train()
    t0 = time.time()
    for batch_idx, x in enumerate(tqdm(loader, mininterval=10, desc='ntp-diag')):
        x = x.to(device) / 100.0  # match pretrain normalization

        if baseline_per_pos is None:
            with torch.no_grad():
                base = ((x[:, :, :-1, :] - x[:, :, 1:, :]) ** 2).mean(dim=(0, 1, 3))
                baseline_per_pos = base.cpu().tolist()
                print(f'[diag] trivial-copy baseline per-position MSE (first batch): {baseline_per_pos}')

        optimizer.zero_grad()
        y = model(x[:, :, :-1, :])
        labels = x[:, :, 1:, :]
        diffs = (y - labels) ** 2
        per_pos = diffs.mean(dim=(0, 1, 3))
        total = per_pos.mean()
        total.backward()
        if args.clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
        optimizer.step()

        if window_per_pos is None:
            window_per_pos = per_pos.detach().clone()
        else:
            window_per_pos += per_pos.detach()
        window_avg += float(total.item())
        window_count += 1
        last_batch_idx = batch_idx

        if window_count >= args.log_window:
            history.append({
                'batch_idx': batch_idx,
                'avg_mse': window_avg / window_count,
                'per_position_mse': (window_per_pos / window_count).cpu().tolist(),
            })
            window_per_pos = None
            window_avg = 0.0
            window_count = 0
            _dump(final=False)

    if window_count > 0:
        history.append({
            'batch_idx': last_batch_idx,
            'avg_mse': window_avg / window_count,
            'per_position_mse': (window_per_pos / window_count).cpu().tolist(),
        })

    _dump(final=True)
    wall = time.time() - t0
    print(f'[diag] wall {wall/60:.1f} min, wrote {out_path}')
    if history:
        print('[diag] final-window per-position MSE:')
        print('  pos: ' + ' '.join(f'{i:6d}' for i in range(len(history[-1]['per_position_mse']))))
        print('  mse: ' + ' '.join(f'{v:6.4f}' for v in history[-1]['per_position_mse']))
        print(f'  baseline: ' + ' '.join(f'{v:6.4f}' for v in baseline_per_pos))

    try:
        ds.db.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
