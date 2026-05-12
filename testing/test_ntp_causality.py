"""
Verify the NTP variant of CBraMod has no time leakage.

For each cut t in [0, patch_num-2]:
  - Take a random input x.
  - Build x' identical for time <= t, scrambled for time > t.
  - Run both through the model.
  - Assert outputs at time <= t are identical.

If any component (pos conv, attention, FFT, etc.) leaks future info into past
positions, the past outputs will differ and this test will fail.

We run the test in BOTH eval and train mode, across multiple seeds, to catch:
  - Mode-dependent leakage (e.g. BatchNorm running stats — not present, but check)
  - Input-dependent leakage (e.g. numerical edge cases in softmax)

Note: in train mode, dropout is active. Dropout is per-position, but to make
identical forward calls deterministic we re-seed torch right before each
forward so the dropout mask is the same for both x and x_perturbed.
"""
import sys
import torch

sys.path.insert(0, '/home/ayf4/cbramod')
from models.cbramod import CBraMod


def run_check(model, x, label, tol=1e-5):
    """Return (max_leak, n_failures) for a single forward sweep."""
    T = x.shape[2]
    max_leak = 0.0
    failures = 0
    # baseline
    torch.manual_seed(12345)
    with torch.no_grad():
        y = model(x)

    for t in range(T - 1):
        x_perturbed = x.clone()
        x_perturbed[:, :, t + 1:, :] = torch.randn_like(x_perturbed[:, :, t + 1:, :])
        # re-seed so dropout masks (if any) are identical to the baseline forward
        torch.manual_seed(12345)
        with torch.no_grad():
            y_perturbed = model(x_perturbed)
        diff = (y[:, :, :t + 1, :] - y_perturbed[:, :, :t + 1, :]).abs().max().item()
        if diff > tol:
            failures += 1
        max_leak = max(max_leak, diff)

    status = 'OK' if max_leak < tol else 'LEAK'
    print(f'  {label:20s}  max|diff|={max_leak:.2e}  [{status}]  fails={failures}/{T-1}')
    return max_leak


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bz, ch, T, P = 2, 19, 29, 200

    print('Building NTP CBraMod...')
    model = CBraMod(in_dim=P, out_dim=P, d_model=P, dim_feedforward=4 * P,
                    seq_len=T, n_layer=12, nhead=8, objective='ntp').to(device)

    worst = 0.0
    for seed in [0, 1, 42, 2026, 9999]:
        torch.manual_seed(seed)
        x = torch.randn(bz, ch, T, P, device=device)

        model.eval()
        worst = max(worst, run_check(model, x, f'seed={seed} eval'))

        model.train()
        worst = max(worst, run_check(model, x, f'seed={seed} train'))

    print('-' * 60)
    print(f'Worst leakage across all (seed, mode) sweeps: {worst:.2e}')
    if worst < 1e-5:
        print('PASS: no time leakage detected.')
    else:
        print('FAIL: time leakage detected.')
        sys.exit(1)


if __name__ == '__main__':
    main()
