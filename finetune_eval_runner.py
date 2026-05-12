"""Finetune-from-pretraining-checkpoint validator.

Stage 0: single benchmark, no warm-down.
Stage 1: optional warm-down on TUEG (1 epoch linear LR decay -> 0), then serially
finetune each benchmark from the warm-down weights (reloading between benchmarks).
"""
import argparse
import json
import os
import time
from argparse import Namespace

import torch

from eval_presets import FINETUNE_DEFAULTS, FINETUNE_EPOCHS, PRESETS, TUEG_LMDB_DIR
from finetune_main import dispatch_finetune, setup_seed
from warmdown import warmdown


def build_params(ckpt_path, benchmark, output_dir, epochs, cuda, save_ckpt, objective):
    if benchmark not in PRESETS:
        raise ValueError('Unknown benchmark {}; known: {}'.format(benchmark, list(PRESETS)))
    preset = PRESETS[benchmark]
    if not preset.get('datasets_dir'):
        raise ValueError(
            "PRESETS['{}']['datasets_dir'] is empty -- fill it in eval_presets.py".format(benchmark)
        )
    merged = dict(FINETUNE_DEFAULTS)
    merged.update(preset)
    merged['downstream_dataset'] = benchmark
    merged['foundation_dir'] = ckpt_path
    merged['use_pretrained_weights'] = True
    merged['model_dir'] = output_dir
    merged['epochs'] = epochs
    merged['cuda'] = cuda
    merged['save_ckpt'] = save_ckpt
    merged['objective'] = objective
    return Namespace(**merged)


def resolve_benchmarks(arg_value):
    if arg_value == 'all':
        names = list(PRESETS.keys())
    else:
        names = [b.strip() for b in arg_value.split(',') if b.strip()]
        unknown = [b for b in names if b not in PRESETS]
        if unknown:
            raise ValueError('Unknown benchmark(s): {}'.format(unknown))
    ready, skipped = [], []
    for b in names:
        if PRESETS[b].get('datasets_dir'):
            ready.append(b)
        else:
            skipped.append(b)
    return ready, skipped


def run_benchmark(ckpt_for_finetune, benchmark, output_dir, epochs, cuda, save_ckpt, objective):
    sub_outdir = os.path.join(output_dir, benchmark)
    os.makedirs(sub_outdir, exist_ok=True)
    params = build_params(ckpt_for_finetune, benchmark, sub_outdir, epochs, cuda, save_ckpt, objective)
    print(f'\n[eval] === benchmark={benchmark}  epochs={epochs}  save_ckpt={save_ckpt} ===')
    print(f'[eval] foundation_dir={ckpt_for_finetune}')
    setup_seed(params.seed)
    t0 = time.time()
    metrics = dispatch_finetune(params)
    wall = time.time() - t0
    return {
        'benchmark': benchmark,
        'epochs': epochs,
        'wall_seconds': wall,
        'wall_minutes': wall / 60.0,
        'metrics': metrics,
    }


def main():
    p = argparse.ArgumentParser(description='Finetune-from-pretraining-checkpoint validator')
    p.add_argument('--ckpt-path', required=True, help='Pretraining checkpoint .pth (model state dict)')
    p.add_argument('--output-dir', required=True,
                   help='Top-level dir; per-benchmark subdirs created inside')
    p.add_argument('--benchmarks', default='all',
                   help='Comma-separated subset of PRESETS, or "all" (default). '
                        'Benchmarks with empty datasets_dir are skipped.')
    p.add_argument('--epochs', type=int, default=FINETUNE_EPOCHS,
                   help=f'Finetune epochs per benchmark (default {FINETUNE_EPOCHS})')
    p.add_argument('--warmdown', dest='warmdown', action='store_true', default=False,
                   help='Run 1-epoch TUEG warm-down before finetune. Off by default '
                        '(empirically no benefit at mid/late-pretrain checkpoints, ~73 min cost on RTX6000).')
    p.add_argument('--no-warmdown', dest='warmdown', action='store_false')
    p.add_argument('--tueg-data-dir', default=TUEG_LMDB_DIR)
    p.add_argument('--objective', choices=('recon', 'ntp'), default='recon',
                   help='Pretraining objective of the input checkpoint (must match)')
    p.add_argument('--need-mask', dest='need_mask', action='store_true', default=True,
                   help='Masked-MSE training during warm-down (default on; auto off for ntp)')
    p.add_argument('--no-need-mask', dest='need_mask', action='store_false')
    p.add_argument('--save-models', dest='save_models', action='store_true', default=False,
                   help='Save per-benchmark .pth at the end (default off; ~533MB per benchmark)')
    p.add_argument('--no-save-models', dest='save_models', action='store_false')
    p.add_argument('--cuda', type=int, default=0)
    p.add_argument('--parent-wandb-run-id', default=None,
                   help='If set, wandb.init(id=..., resume="allow") and write summary keys '
                        'like eval/<bench>/<metric>@e<pretrain_epoch>')
    p.add_argument('--wandb-project', default='cbramod-pretrain-tueg')
    p.add_argument('--pretrain-epoch', type=int, default=None,
                   help='Pretrain epoch this checkpoint corresponds to; used to namespace '
                        'wandb summary keys (required when --parent-wandb-run-id is set)')
    args = p.parse_args()

    if not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(args.ckpt_path)
    os.makedirs(args.output_dir, exist_ok=True)

    benchmarks, skipped = resolve_benchmarks(args.benchmarks)
    print(f'[eval] epochs={args.epochs}  save_models={args.save_models}  warmdown={args.warmdown}')
    print(f'[eval] benchmarks={benchmarks}  skipped(empty datasets_dir)={skipped}')

    # While the parent pretrain run is still active, writes from a second
    # wandb.init(resume='allow') process are silently dropped by the backend
    # (concurrent-writer is undefined in wandb 0.26). But evals that land AFTER
    # pretrain calls wandb.finish() — notably the epoch-40 trigger and anything
    # in the SLURM queue when pretrain ends — DO get a clean resume and their
    # writes stick. So we still attempt the resume here; for the epochs whose
    # writes got dropped, sync_evals_to_wandb.py backfills from
    # eval_epoch*/results.json.
    wandb_run = None
    if args.parent_wandb_run_id:
        if args.pretrain_epoch is None:
            raise ValueError('--pretrain-epoch is required when --parent-wandb-run-id is set')
        try:
            import wandb as _wandb
            wandb_run = _wandb.init(
                project=args.wandb_project,
                id=args.parent_wandb_run_id,
                resume='allow',
            )
            _wandb.define_metric('pretrain_epoch')
            _wandb.define_metric('eval/*', step_metric='pretrain_epoch')
            print(f'[eval] wandb resumed run id={wandb_run.id} project={args.wandb_project}')
        except Exception as e:
            print(f'[eval] WARNING: wandb.init failed ({type(e).__name__}: {e}); continuing without wandb')
            wandb_run = None

    torch.cuda.set_device(args.cuda)

    overall_t0 = time.time()
    aggregate = {
        'ckpt_path': os.path.abspath(args.ckpt_path),
        'epochs': args.epochs,
        'save_models': args.save_models,
        'warmdown': args.warmdown,
        'benchmarks_skipped': skipped,
        'warmdown_info': None,
        'results': [],
    }

    epoch_suffix = f'@e{args.pretrain_epoch}' if args.pretrain_epoch is not None else ''

    # Warm-down (optional). Produces output_dir/warmdown.pth used for all finetunes.
    if args.warmdown:
        warmdown_ckpt, wd_info = warmdown(
            ckpt_path=args.ckpt_path,
            output_dir=args.output_dir,
            tueg_data_dir=args.tueg_data_dir,
            cuda=args.cuda,
            objective=args.objective,
            need_mask=args.need_mask,
        )
        aggregate['warmdown_info'] = wd_info
        finetune_init = warmdown_ckpt
    else:
        finetune_init = args.ckpt_path

    for b in benchmarks:
        res = run_benchmark(
            ckpt_for_finetune=finetune_init,
            benchmark=b,
            output_dir=args.output_dir,
            epochs=args.epochs,
            cuda=args.cuda,
            save_ckpt=args.save_models,
            objective=args.objective,
        )
        aggregate['results'].append(res)
        # Also drop a per-benchmark file in case anything truncates the aggregate.
        with open(os.path.join(args.output_dir, b, 'result.json'), 'w') as f:
            json.dump(res, f, indent=2)
        print(f'[eval] {b}: wall {res["wall_minutes"]:.2f} min  metrics={res["metrics"]}')
        if wandb_run is not None:
            log_payload = {f'eval/{b}/{k}': v for k, v in res['metrics'].items()}
            log_payload[f'eval/{b}/wall_minutes'] = res['wall_minutes']
            log_payload['pretrain_epoch'] = args.pretrain_epoch
            wandb_run.log(log_payload)

    aggregate['total_wall_seconds'] = time.time() - overall_t0
    aggregate['total_wall_minutes'] = aggregate['total_wall_seconds'] / 60.0

    out_json = os.path.join(args.output_dir, 'results.json')
    with open(out_json, 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f'\n[eval] wrote aggregate {out_json}')
    print(f'[eval] TOTAL wall {aggregate["total_wall_minutes"]:.2f} min across {len(benchmarks)} benchmarks')

    if wandb_run is not None:
        wandb_run.finish(quiet=True)
        # Sentinel so sync_evals_to_wandb.py doesn't double-log this epoch.
        # Only created when wandb.init succeeded AND we got here — the writes
        # have been pushed to the wandb service.
        with open(os.path.join(args.output_dir, '.wandb_logged'), 'w') as f:
            f.write(f'parent_run_id={args.parent_wandb_run_id}\npretrain_epoch={args.pretrain_epoch}\n')



if __name__ == '__main__':
    main()
