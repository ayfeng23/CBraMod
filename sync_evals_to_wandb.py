"""Post-hoc backfill of eval results into a finished pretrain wandb run.

Reads <parent-run-dir>/eval_epoch*/results.json and logs each benchmark's metrics
as one wandb.log() call indexed by pretrain_epoch. Run AFTER the pretrain finishes
so the wandb run is no longer owned by an active writer (this avoids the
concurrent-writer issue where summary updates get silently dropped).

Usage:
    python sync_evals_to_wandb.py \\
        --parent-run-dir /path/to/cbramod_pretrain/40ep_recon_11417090 \\
        --wandb-run-id gz3eh12z \\
        --wandb-project cbramod-pretrain-tueg
"""
import argparse
import glob
import json
import os
import re

import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent-run-dir', required=True)
    p.add_argument('--wandb-run-id', required=True)
    p.add_argument('--wandb-project', default='cbramod-pretrain-tueg')
    args = p.parse_args()

    pattern = os.path.join(args.parent_run_dir, 'eval_epoch*', 'results.json')
    paths = sorted(glob.glob(pattern), key=lambda p: int(re.search(r'eval_epoch(\d+)', p).group(1)))
    if not paths:
        raise SystemExit(f'no eval results found under {pattern}')

    print(f'[sync] found {len(paths)} eval_epoch dirs under {args.parent_run_dir}')

    run = wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume='must')
    wandb.define_metric('pretrain_epoch')
    wandb.define_metric('eval/*', step_metric='pretrain_epoch')

    for path in paths:
        epoch = int(re.search(r'eval_epoch(\d+)', path).group(1))
        sentinel = os.path.join(os.path.dirname(path), '.wandb_logged')
        if os.path.exists(sentinel):
            print(f'[sync] epoch={epoch} SKIP (already logged live via {sentinel})')
            continue
        with open(path) as f:
            agg = json.load(f)
        log_payload = {'pretrain_epoch': epoch}
        for res in agg.get('results', []):
            b = res['benchmark']
            for k, v in res['metrics'].items():
                log_payload[f'eval/{b}/{k}'] = v
            log_payload[f'eval/{b}/wall_minutes'] = res['wall_minutes']
        run.log(log_payload)
        with open(sentinel, 'w') as f:
            f.write(f'parent_run_id={args.wandb_run_id}\npretrain_epoch={epoch}\nlogged_via=sync\n')
        print(f'[sync] epoch={epoch} logged {len(log_payload)-1} keys')

    run.finish()
    print('[sync] done')


if __name__ == '__main__':
    main()
