import argparse
import random

import numpy as np
import torch

from datasets import faced_dataset, seedv_dataset, physio_dataset, shu_dataset, isruc_dataset, chb_dataset, \
    speech_dataset, mumtaz_dataset, seedvig_dataset, stress_dataset, tuev_dataset, tuab_dataset, bciciv2a_dataset
from finetune_trainer import Trainer
from models import model_for_faced, model_for_seedv, model_for_physio, model_for_shu, model_for_isruc, model_for_chb, \
    model_for_speech, model_for_mumtaz, model_for_seedvig, model_for_stress, model_for_tuev, model_for_tuab, \
    model_for_bciciv2a


def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=1, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, '
                             'all_patch_reps_onelayer, avgpooling_patch_reps]')
    # all_patch_reps: use all patch features with a three-layer classifier;
    # all_patch_reps_twolayer: use all patch features with a two-layer classifier;
    # all_patch_reps_onelayer: use all patch features with a one-layer classifier;
    # avgpooling_patch_reps: use average pooling for patch features;

    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='MentalArithmetic',
                        help='[FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, CHB-MIT, BCIC2020-3, Mumtaz2016, '
                             'SEED-VIG, MentalArithmetic, TUEV, TUAB, BCIC-IV-2a]')
    parser.add_argument('--datasets_dir', type=str,
                        default='/data/datasets/BigDownstream/mental-arithmetic/processed',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='/data/wjq/models_weights/Big/BigFaced', help='model_dir')
    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--multi_lr', type=bool, default=True,
                        help='multi_lr')  # set different learning rates for different modules
    parser.add_argument('--frozen', type=bool,
                        default=False, help='frozen')
    parser.add_argument('--use_pretrained_weights', type=bool,
                        default=True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        default='pretrained_weights/pretrained_weights.pth',
                        help='foundation_dir')

    params = parser.parse_args()
    print(params)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    dispatch_finetune(params)
    print('Done!!!!!')


# (downstream_dataset, dataset_module, model_module, train_method_name)
# More readable than prev version
_DATASET_DISPATCH = {
    'FACED':            (faced_dataset,    model_for_faced,    'train_for_multiclass'),
    'SEED-V':           (seedv_dataset,    model_for_seedv,    'train_for_multiclass'),
    'PhysioNet-MI':     (physio_dataset,   model_for_physio,   'train_for_multiclass'),
    'SHU-MI':           (shu_dataset,      model_for_shu,      'train_for_binaryclass'),
    'ISRUC':            (isruc_dataset,    model_for_isruc,    'train_for_multiclass'),
    'CHB-MIT':          (chb_dataset,      model_for_chb,      'train_for_binaryclass'),
    'BCIC2020-3':       (speech_dataset,   model_for_speech,   'train_for_multiclass'),
    'Mumtaz2016':       (mumtaz_dataset,   model_for_mumtaz,   'train_for_binaryclass'),
    'SEED-VIG':         (seedvig_dataset,  model_for_seedvig,  'train_for_regression'),
    'MentalArithmetic': (stress_dataset,   model_for_stress,   'train_for_binaryclass'),
    'TUEV':             (tuev_dataset,     model_for_tuev,     'train_for_multiclass'),
    'TUAB':             (tuab_dataset,     model_for_tuab,     'train_for_binaryclass'),
    'BCIC-IV-2a':       (bciciv2a_dataset, model_for_bciciv2a, 'train_for_multiclass'),
}


def dispatch_finetune(params):
    """Build dataset + model + Trainer for params.downstream_dataset and run training.
    Returns dict of final test metrics from the underlying train_for_* method.
    Caller is responsible for setup_seed() and torch.cuda.set_device()."""
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    if params.downstream_dataset not in _DATASET_DISPATCH:
        raise ValueError('Unknown downstream_dataset: {}'.format(params.downstream_dataset))
    dataset_module, model_module, train_method = _DATASET_DISPATCH[params.downstream_dataset]
    load_dataset = dataset_module.LoadDataset(params)
    data_loader = load_dataset.get_data_loader()
    model = model_module.Model(params)
    t = Trainer(params, data_loader, model)
    return getattr(t, train_method)()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()