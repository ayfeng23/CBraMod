"""TUEG preprocessing -- one task processes one top-level shard (e.g. 000/).

Mirrors preprocessing_tueg_for_pretraining.py one-to-one (resample 200 Hz,
bandpass 0.3-75 Hz, notch 60 Hz, drop |x|>=100uV windows). Writes accepted
30-s windows to {output_root}/shard_{shard}.lmdb.
"""

import argparse
import glob
import os
import pickle
import random
import sys
import time

import lmdb
import mne
import numpy as np
from tqdm import tqdm

CHANS = {
    '01_tcp_ar': [
        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
        'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
        'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
        'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
    ],
    '02_tcp_le': [
        'EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE',
        'EEG C4-LE', 'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE',
        'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE',
        'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE',
    ],
    '03_tcp_ar_a': [
        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
        'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
        'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
        'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
    ],
}
MAP_SIZE = 64 * 1024 ** 3  # 64 GiB sparse; per-shard LMDB is ~10-20 GB actual


def montage(path):
    for tag in ('02_tcp_le', '03_tcp_ar_a', '01_tcp_ar'):
        if tag in path:
            return tag
    return None


def process(fp):
    """Filter+window one EDF. Returns list of (key, ndarray) for accepted windows."""
    raw = mne.io.read_raw_edf(fp, preload=False, verbose='ERROR')
    m = montage(fp)
    if m is None:
        return []
    chans = CHANS[m]
    if not all(c in raw.info['ch_names'] for c in chans):
        return []
    raw.pick(chans)                  # metadata-only; data still on disk
    raw.load_data(verbose='ERROR')   # load only the 19 picked channels
    raw.resample(200, verbose='ERROR')
    raw.filter(0.3, 75, verbose='ERROR')
    raw.notch_filter(60, verbose='ERROR')

    x = raw.get_data().T * 1e6  # V -> uV (matches original script's to_data_frame() units)
    n, c = x.shape
    if n < 300 * 200:
        return []
    a = n % (30 * 200)
    x = x[60 * 200:n - (a + 60 * 200), :]
    if x.shape[0] < 30 * 200:
        return []
    x = x.reshape(-1, 30, 200, c).transpose(0, 3, 1, 2)
    name = os.path.basename(fp)[:-4]
    return [(f'{name}_{i}', s) for i, s in enumerate(x) if np.max(np.abs(s)) < 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', required=True, help='top-level subdir, e.g. "000"')
    ap.add_argument('--input_root', required=True)
    ap.add_argument('--output_root', required=True)
    args = ap.parse_args()

    random.seed(1)
    np.random.seed(1)

    shard_dir = os.path.join(args.input_root, args.shard)
    if not os.path.isdir(shard_dir):
        sys.exit(f'no such dir: {shard_dir}')

    edfs = sorted(glob.glob(os.path.join(shard_dir, '**', '*.edf'), recursive=True))
    random.shuffle(edfs)

    os.makedirs(args.output_root, exist_ok=True)
    out_path = os.path.join(args.output_root, f'shard_{args.shard}.lmdb')
    db = lmdb.open(out_path, map_size=MAP_SIZE)

    all_keys = []
    t0 = time.time()
    for fp in tqdm(edfs, desc=f'shard {args.shard}'):
        try:
            samples = process(fp)
        except Exception as e:
            print(f'FAIL {fp}: {e}', file=sys.stderr)
            continue
        if not samples:
            continue
        with db.begin(write=True) as txn:
            for k, s in samples:
                txn.put(k.encode(), pickle.dumps(s))
                all_keys.append(k)

    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(all_keys))
    db.close()

    print(f'shard {args.shard}: {len(all_keys)} windows from {len(edfs)} EDFs '
          f'in {time.time() - t0:.1f}s -> {out_path}')


if __name__ == '__main__':
    main()
