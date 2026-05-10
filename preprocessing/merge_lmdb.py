"""Merge per-shard LMDBs (shard_*.lmdb) into a single LMDB.

Commits in small batches so peak RAM stays bounded. Tracks completed shards
in a sidecar `.progress` file next to the output so a kill is resumable.
"""

import argparse
import glob
import os
import pickle

import lmdb
from tqdm import tqdm

MAP_SIZE = 4 * 1024 ** 4  # 4 TiB sparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards_dir', required=True, help='dir containing shard_*.lmdb')
    ap.add_argument('--out', required=True, help='path to merged LMDB env')
    ap.add_argument('--batch', type=int, default=500,
                    help='keys per commit (~900 KB/key -> ~450 MB peak per txn)')
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.shards_dir, 'shard_*.lmdb')))
    if not shards:
        raise SystemExit(f'no shard_*.lmdb found in {args.shards_dir}')

    progress_file = args.out + '.progress'
    done_paths = set()
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            done_paths = {ln.strip() for ln in f if ln.strip()}
        print(f'resuming: {len(done_paths)}/{len(shards)} shards already merged')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out_db = lmdb.open(args.out, map_size=MAP_SIZE)
    all_keys = []
    incomplete = []

    for path in tqdm(shards):
        env = lmdb.open(path, readonly=True, lock=False)
        with env.begin() as src:
            raw = src.get(b'__keys__')
            if raw is None:
                incomplete.append(path)
                env.close()
                continue
            keys = pickle.loads(raw)

            if path not in done_paths:
                for start in range(0, len(keys), args.batch):
                    batch = keys[start:start + args.batch]
                    with out_db.begin(write=True) as dst:
                        for k in batch:
                            bk = k.encode()
                            dst.put(bk, src.get(bk))
                with open(progress_file, 'a') as f:
                    f.write(path + '\n')
        env.close()
        all_keys.extend(keys)

    with out_db.begin(write=True) as dst:
        dst.put(b'__keys__', pickle.dumps(all_keys))
    out_db.close()

    if os.path.exists(progress_file):
        os.remove(progress_file)
    print(f'done: {len(all_keys)} keys merged into {args.out}')
    if incomplete:
        print(f'WARNING: {len(incomplete)} shards missing __keys__:')
        for p in incomplete:
            print(f'  {p}')


if __name__ == '__main__':
    main()
