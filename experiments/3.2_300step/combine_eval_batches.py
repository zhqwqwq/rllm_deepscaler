#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recursively merge all '*__shards' directories under current working directory.

Rules:
- For each shards dir like: /path/to/<base>__shards
  Merge batch_*.parquet in numeric order into: /path/to/<base>.parquet
- Skip if no shards found.
- Skip if output exists and is newer than all shards (incremental-safe).
- Atomic write: write to <output>.tmp then os.replace to <output>.

Usage:
  python merge_all_shards.py   # no arguments
"""

# ROOT="data/Llama3_GSM8K_rollout_16"
# ROOT="data/Llama3_GSM8K_run2_ckpt300_40rollouts"
# ROOT="/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/countdown_run1/10.13_countdown_40960*5"
ROOT="/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_300_47rollout"

import os
import re
import glob
import sys
from typing import List, Tuple, Optional

import pyarrow as pa
import pyarrow.parquet as pq

BASENAME_PATTERN = re.compile(r"batch_(\d+)\.parquet$")

def find_shard_dirs(root: str = ".") -> List[str]:
    shard_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        # only consider leaf folders that end with __shards
        if dirpath.endswith("__shards"):
            shard_dirs.append(dirpath)
    shard_dirs.sort()
    return shard_dirs

def list_shards_numeric(shard_dir: str) -> List[str]:
    files = glob.glob(os.path.join(shard_dir, "batch_*.parquet"))
    parsed: List[Tuple[int, str]] = []
    for f in files:
        m = BASENAME_PATTERN.search(os.path.basename(f))
        if m:
            parsed.append((int(m.group(1)), f))
    parsed.sort(key=lambda x: x[0])
    return [p[1] for p in parsed]

def base_output_from_shard_dir(shard_dir: str) -> str:
    """
    .../foo/bar/<base>__shards  ->  .../foo/bar/<base>.parquet
    """
    parent = os.path.dirname(shard_dir)
    base_shards = os.path.basename(shard_dir)         # "<base>__shards"
    assert base_shards.endswith("__shards")
    base = base_shards[:-len("__shards")]             # "<base>"
    return os.path.join(parent, base + ".parquet")

def newest_mtime(paths: List[str]) -> float:
    return max(os.path.getmtime(p) for p in paths)

def atomic_replace(src: str, dst: str):
    os.replace(src, dst)

def merge_one(shard_dir: str) -> Optional[str]:
    shards = list_shards_numeric(shard_dir)
    if not shards:
        print(f"[SKIP] No shards in {shard_dir}")
        return None

    out_path = base_output_from_shard_dir(shard_dir)
    tmp_out = out_path + ".tmp"

    # incremental-safe: if output exists and is newer than all shards, skip
    try:
        out_mtime = os.path.getmtime(out_path)
        shards_mtime = newest_mtime(shards)
        if out_mtime >= shards_mtime:
            print(f"[SKIP] Up-to-date: {out_path}")
            return out_path
    except FileNotFoundError:
        pass

    first_pf = pq.ParquetFile(shards[0])
    schema = first_pf.schema_arrow
    writer = pq.ParquetWriter(tmp_out, schema=schema, compression=None)

    total_rows = 0
    try:
        for i, f in enumerate(shards, 1):
            pf = pq.ParquetFile(f)

            # schema check
            if pf.schema_arrow != schema:
                writer.close()
                try:
                    if os.path.exists(tmp_out):
                        os.remove(tmp_out)
                except Exception:
                    pass
                raise RuntimeError(
                    f"[ERROR] Schema mismatch at {f}. "
                    "Please normalize schemas/columns before merging."
                )

            file_rows = 0
            for rg_idx in range(pf.num_row_groups):
                table = pf.read_row_group(rg_idx)
                file_rows += table.num_rows
                writer.write_table(table)

            total_rows += file_rows
            print(f"[MERGE] {os.path.basename(shard_dir)} "
                  f"[{i}/{len(shards)}] +{file_rows} rows (total={total_rows})")

    except Exception:
        writer.close()
        try:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass
        raise

    writer.close()
    atomic_replace(tmp_out, out_path)
    print(f"[DONE ] Merged {len(shards)} shard(s), total {total_rows} rows → {out_path}")
    return out_path

def main():
    root = ROOT
    shard_dirs = find_shard_dirs(root)
    if not shard_dirs:
        print("[INFO] No '*__shards' directories found. Nothing to do.")
        return

    print(f"[INFO] Found {len(shard_dirs)} shard dir(s). Start merging...\n")
    ok, fail = 0, 0
    for sd in shard_dirs:
        base = os.path.basename(sd)
        try:
            res = merge_one(sd)
            if res is not None:
                ok += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL ] {base}: {e}\n")

    print(f"\n[SUMMARY] merged_ok={ok}, failed={fail}, scanned={len(shard_dirs)}")

if __name__ == "__main__":
    main()
