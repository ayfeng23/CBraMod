import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.pretraining_dataset import PretrainingDataset
from models.cbramod import CBraMod
from pretrain_trainer import Trainer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='EEG Foundation Model')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
    parser.add_argument('--parallel', type=bool, default=False, help='parallel')
    parser.add_argument('--objective', type=str, default="recon", help='ntp or recon')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight_decay')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR',
                        help='lr_scheduler: CosineAnnealingLR, ExponentialLR, StepLR, MultiStepLR, CyclicLR')

    # parser.add_argument('--project_mode', type=str, default='cnn', help='project_mode')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--in_dim', type=int, default=200, help='in_dim')
    parser.add_argument('--out_dim', type=int, default=200, help='out_dim')
    parser.add_argument('--d_model', type=int, default=200, help='d_model')
    parser.add_argument('--dim_feedforward', type=int, default=800, help='dim_feedforward')
    parser.add_argument('--seq_len', type=int, default=30, help='seq_len')
    parser.add_argument('--n_layer', type=int, default=12, help='n_layer')
    parser.add_argument('--nhead', type=int, default=8, help='nhead')
    parser.add_argument('--need_mask', type=bool, default=True, help='need_mask')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask_ratio')

    parser.add_argument('--dataset_dir', type=str, default='dataset_dir',
                        help='dataset_dir')
    parser.add_argument('--model_dir',   type=str,   default='model_dir', help='model_dir')
    parser.add_argument('--wandb', action='store_true', help='enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='cbramod-pretrain-tueg')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='wandb run name (default: wandb auto / SLURM job id)')
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint to warm-start from (model state_dict). '
                             'If a sibling <ckpt>.full.pth exists, optimizer/scheduler/epoch are restored too.')
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='manually specify the epoch the checkpoint was saved at. '
                             'Used only when no .full.pth companion exists: training resumes from this epoch, '
                             'optimizer starts fresh, scheduler is fast-forwarded to match LR.')
    parser.add_argument('--resume_lr', type=float, default=None,
                        help='override LR after warm-start (useful when only model weights are loaded). '
                             'If unset and full state was restored, the scheduler keeps its course.')
    params = parser.parse_args()
    print(params)
    setup_seed(params.seed)
    pretrained_dataset = PretrainingDataset(dataset_dir=params.dataset_dir)
    print(len(pretrained_dataset))
    data_loader = DataLoader(
        pretrained_dataset,
        batch_size=params.batch_size,
        num_workers=8,
        shuffle=True,
    )
    if params.objective == "ntp":
        params.need_mask = False
    
    print(f"Using {params.objective} objective {'with' if params.need_mask else 'without'} masking")
    model = CBraMod(
        params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len, params.n_layer,
        params.nhead, params.objective
    )
    trainer = Trainer(params, data_loader, model)
    trainer.train()
    pretrained_dataset.db.close()


if __name__ == '__main__':
    main()
