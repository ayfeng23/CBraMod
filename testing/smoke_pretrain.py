"""CPU smoke test: dataset -> model -> forward -> backward, one batch.

Validates the merged TUEG LMDB feeds CBraMod end-to-end. No GPU, no real
training -- just confirms shapes, loss compute, and grad flow work.
"""

import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/ayf4/cbramod')
from datasets.pretraining_dataset import PretrainingDataset  # noqa: E402
from models.cbramod import CBraMod                           # noqa: E402
from utils.util import generate_mask                         # noqa: E402

DATA = '/home/ayf4/scratch_pi_zf59/ayf4/data/tueg_v2.0.1_lmdb'
device = torch.device('cpu')

t0 = time.time()
ds = PretrainingDataset(DATA)
print(f'dataset: {len(ds):,} samples (loaded in {time.time() - t0:.1f}s)')

dl = DataLoader(ds, batch_size=2, num_workers=0, shuffle=True)
t0 = time.time()
batch = next(iter(dl))
print(f'batch: {tuple(batch.shape)} {batch.dtype} '
      f'range=[{batch.min():.1f},{batch.max():.1f}] uV  ({time.time() - t0:.1f}s)')

model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                seq_len=30, n_layer=12, nhead=8).to(device)
nparams = sum(p.numel() for p in model.parameters())
print(f'model params: {nparams / 1e6:.1f}M')

x = batch.to(device) / 100
bz, ch_num, patch_num, _ = x.shape
mask = generate_mask(bz, ch_num, patch_num, mask_ratio=0.5, device=device)

t0 = time.time()
y = model(x, mask=mask)
print(f'forward: in={tuple(x.shape)} out={tuple(y.shape)} ({time.time() - t0:.1f}s)')

loss = nn.MSELoss()(y[mask == 1], x[mask == 1])
print(f'loss: {loss.item():.6f}')

t0 = time.time()
loss.backward()
gnorm = sum((p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)) ** 0.5
print(f'backward: grad-norm={gnorm:.4f} ({time.time() - t0:.1f}s)')

print('\nSMOKE TEST PASSED')
