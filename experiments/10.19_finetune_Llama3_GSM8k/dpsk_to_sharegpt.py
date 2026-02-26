#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge two GSM8K rollout JSONL files and convert to ShareGPT JSONL.
- Input record example:
  {
    "row_idx": 15,
    "sample_idx": 0,
    "question": "...",
    "cot": "...",
    "final": "...",
    ...
  }
- Output (per line):
  {"conversations": [
      {"from": "human", "value": <question>},
      {"from": "gpt",   "value": "<think>" + cot + "</think>\n\n" + final}
  ]}
Notes:
- Keep <think> blocks as-is (no stripping).
- If "cot" or "final" is missing, degrade gracefully.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Dict, Any
from transformers import AutoTokenizer

# def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             yield json.loads(line)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            print(i)
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Error] {path} line {i}: {e}")
                print(line[:200])
                raise

def build_reply(cot: str | None, final: str | None) -> str:
    cot = (cot or "").strip()
    final = (final or "").strip()

    has_cot = len(cot) > 0
    has_final = len(final) > 0

    if has_cot and has_final:
        return f"<think>\n{cot}\n</think>\n\n{final}"
    elif has_cot and not has_final:
        return f"<think>\n{cot}\n</think>"
    elif not has_cot and has_final:
        return final
    else:
        # 极端情况：都没有，返回空串（上游会跳过）
        return ""

def to_sharegpt_item(question: str, reply: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt",   "value": reply},
        ]
    }

max_token = 4096

def main():
    # parser = argparse.ArgumentParser(description="Convert DeepSeek GSM8K rollout JSONL to ShareGPT JSONL.")
    # parser.add_argument("--in1", required=True, help="Path to first JSONL file.")
    # parser.add_argument("--in2", required=True, help="Path to second JSONL file.")
    # parser.add_argument("--out", required=True, help="Path to output ShareGPT JSONL file.")
    # args = parser.parse_args()

    p1 = Path("/cephfs/zhanghuaqing/RL/rllm/data/10.19_dpsk_gsm_rollout/gsm8k_train_sample00.jsonl")
    p2 = Path("/cephfs/zhanghuaqing/RL/rllm/data/10.19_dpsk_gsm_rollout/gsm8k_train_sample01.jsonl")
    # p3 = Path("/cephfs/zhanghuaqing/RL/rllm/data/10.19_dpsk_gsm_rollout/gsm8k_train_sample02.jsonl")
    out_path = Path(f"/cephfs/zhanghuaqing/RL/rllm/data/10.19_dpsk_gsm_rollout/gsm8k_train_sharegpt_16rollouts_length{max_token}.jsonl")
    
    tokenizer = AutoTokenizer.from_pretrained("/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/finetune_Llama3B/checkpoint-3054", use_fast=True)

    all_rollouts = []
    for idx in range(16):
        path  = Path(f"/cephfs/zhanghuaqing/RL/rllm/data/10.19_dpsk_gsm_rollout/gsm8k_train_sample{idx:02d}.jsonl")
        print(f"Reading rollouts from: {path}")
        all_rollouts += list(read_jsonl(path))

    n_in, n_out = 0, 0
    n_filtered = 0
    n_no_question = 0
    n_no_reply = 0
    with out_path.open("w", encoding="utf-8") as fout:
        # for rec in list(read_jsonl(p1)) + list(read_jsonl(p2)) + list(read_jsonl(p3)):
        # for rec in list(read_jsonl(p1)) + list(read_jsonl(p2)):
        for rec in all_rollouts:
            n_in += 1
            question = (rec.get("question") or "").strip()
            if not question:
                # 有些数据可能把题目放在别的键，这里可以按需补充回退键名
                n_no_question += 1
                continue

            cot = rec.get("cot")
            final = rec.get("final")
            reply = build_reply(cot, final)
            
            token_count = len(tokenizer.encode(reply, add_special_tokens=False))
            if token_count > max_token:
                n_filtered += 1
                continue
            
            if not reply:
                n_no_reply += 1
                continue

            item = to_sharegpt_item(question, reply)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Read: {n_in} records")
    print(f"Wrote: {n_out} ShareGPT samples")
    print(f"Filtered (too long): {n_filtered} samples")
    print(f"No question: {n_no_question}")
    print(f"No reply: {n_no_reply}")
    print(f"Output: {out_path}")

if __name__ == "__main__":
    main()
