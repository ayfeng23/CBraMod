"""Per-benchmark preset overrides for finetune-from-pretraining-checkpoint validation.

Defaults mirror finetune_main.py argparse defaults; PRESETS overrides them per benchmark.
For each benchmark, fill datasets_dir (and any other args your existing finetune workflow uses).
The runner skips benchmarks whose datasets_dir is missing/empty so partial filling is fine."""

# Mirror of finetune_main.py argparse defaults at the time of writing.
# Update if finetune_main defaults change.
FINETUNE_DEFAULTS = {
    'seed': 3407,
    'cuda': 0,  # eval runner is given one GPU by SLURM
    'epochs': 50,
    'batch_size': 64,
    'lr': 1e-4,
    'weight_decay': 5e-2,
    'optimizer': 'AdamW',
    'clip_value': 1.0,
    'dropout': 0.1,
    'classifier': 'all_patch_reps',
    'downstream_dataset': 'MentalArithmetic',
    'datasets_dir': '',
    'num_of_classes': 2,
    'model_dir': '',
    'num_workers': 16,
    'label_smoothing': 0.1,
    'multi_lr': True,
    'frozen': False,
    'use_pretrained_weights': True,
    'foundation_dir': '',
    # eval-only: when False, finetune_trainer skips writing the per-benchmark .pth file
    # (the runner enables this for intermediate evals to avoid 533MB-per-benchmark disk bloat).
    'save_ckpt': True,
    # Must match the pretraining objective the foundation_dir was trained with;
    # changes pos-encoding padding and proj_in batching semantics inside CBraMod.
    'objective': 'recon',
}

FINETUNE_EPOCHS = 50

# Path to pretrained TUEG lmdb used for warm-down.
TUEG_LMDB_DIR = '/home/ayf4/scratch_pi_zf59/ayf4/data/tueg_v2.0.1_lmdb'

PRESETS = {
    'FACED': {
        # lmdb dir (data.mdb/lock.mdb) consumed directly by datasets/faced_dataset.py
        'datasets_dir': '/home/ayf4/scratch_pi_zf59/ayf4/data/faced_processed',
        'num_of_classes': 9,
    },
    'PhysioNet-MI': {
        # lmdb dir consumed directly by datasets/physio_dataset.py
        'datasets_dir': '/home/ayf4/scratch_pi_zf59/ayf4/data/physio_mi_processed_average',
        'num_of_classes': 4,
    },
    'TUEV': {
        # tuev_dataset.py wants processed_train/processed_eval/processed_test as subdirs.
        # preprocessing_tuev.py nests those under .../tuev_v2.0.1_processed/processed/
        'datasets_dir': '/home/ayf4/scratch_pi_zf59/ayf4/data/tuev_v2.0.1_processed/processed',
        'num_of_classes': 6,
    },
    'MentalArithmetic': {
        # PhysioNet eegmat 1.0.0; lmdb produced by preprocessing/preprocessing_stress.py
        'datasets_dir': '/home/ayf4/scratch_pi_zf59/ayf4/data/mental_arithmetic/processed',
        'num_of_classes': 2,
    },
}
