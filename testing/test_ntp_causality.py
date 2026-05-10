"""
Verify the NTP variant of CBraMod has no time leakage.

For each cut t in [0, patch_num-2]:
  - Take a random input x.
  - Build x' identical for time <= t, scrambled for time > t.
  - Run both through the model in eval mode.
  - Assert outputs at time <= t are identical.

If any component (pos conv, attention, FFT, etc.) leaks future info into past
positions, the past outputs will differ and this test will fail.

Run:
    cd /home/ayf4/cbramod
    python -m testing.test_ntp_causality
"""
import sys
import torch

sys.path.insert(0, '/home/ayf4/cbramod')
from models.cbramod import CBraMod


def main():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bz, ch, T, P = 2, 19, 29, 200
    model = CBraMod(in_dim=P, out_dim=P, d_model=P, dim_feedforward=4 * P,
                    seq_len=T, n_layer=12, nhead=8, objective='ntp').to(device).eval()

    x = torch.randn(bz, ch, T, P, device=device)

    # First forward: baseline
    with torch.no_grad():
        y = model(x)

    max_leak = 0.0
    for t in range(T - 1):
        x_perturbed = x.clone()
        # Scramble the future (everything strictly after time t)
        x_perturbed[:, :, t + 1:, :] = torch.randn_like(x_perturbed[:, :, t + 1:, :])

        with torch.no_grad():
            y_perturbed = model(x_perturbed)

        diff_past = (y[:, :, :t + 1, :] - y_perturbed[:, :, :t + 1, :]).abs().max().item()
        max_leak = max(max_leak, diff_past)
        status = 'OK ' if diff_past < 1e-5 else 'LEAK'
        print(f'  t={t:2d}  max|past diff|={diff_past:.2e}  [{status}]')

    print('-' * 50)
    print(f'Worst leakage across all cuts: {max_leak:.2e}')
    if max_leak < 1e-5:
        print('PASS: no time leakage detected.')
    else:
        print('FAIL: time leakage detected.')
        sys.exit(1)


if __name__ == '__main__':
    main()
