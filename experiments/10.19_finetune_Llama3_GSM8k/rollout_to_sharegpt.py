#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a rollout Parquet file into ShareGPT JSONL (keep <think>...</think>).
- Input schema example:
  columns: ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info', 'responses']
  - prompt: either a string, or a list/array of {"role": "...", "content": "..."}
  - responses: array-like with multiple strings
- Output: one ShareGPT-style sample per response:
  {"conversations": [{"from":"human","value":<prompt>}, {"from":"gpt","value":<response>}]}
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, List

import pandas as pd
import numpy as np


def is_nan_like(x: Any) -> bool:
    """Robust NaN / None check for mixed types."""
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return False


def extract_prompt(prompt_field: Any) -> str:
    """
    Accepts either:
      - str
      - list / np.ndarray of dicts with {"role", "content"}
    Returns a single string prompt for ShareGPT "human" turn.
    """
    if isinstance(prompt_field, str):
        return prompt_field.strip()

    if isinstance(prompt_field, np.ndarray):
        prompt_field = prompt_field.tolist()

    if isinstance(prompt_field, (list, tuple)):
        msgs = []
        for m in prompt_field:
            try:
                role = m.get("role", "").lower()
                content = m.get("content", "")
            except AttributeError:
                role, content = "", str(m)
            if role in ("user", "system", "") and isinstance(content, str) and content.strip():
                msgs.append(content.strip())
        if msgs:
            return "\n\n".join(msgs)
        try:
            return json.dumps(prompt_field, ensure_ascii=False)
        except Exception:
            return str(prompt_field)

    return str(prompt_field)


def ensure_iterable_responses(responses_field: Any) -> List[str]:
    """Normalize 'responses' into a list of strings."""
    if is_nan_like(responses_field):
        return []
    if isinstance(responses_field, str):
        return [responses_field]
    if isinstance(responses_field, np.ndarray):
        responses_field = responses_field.tolist()
    if isinstance(responses_field, (list, tuple)):
        return [str(r) for r in responses_field if not is_nan_like(r)]
    return [str(responses_field)]


def build_sharegpt_sample(prompt: str, response: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": response},
        ]
    }

def replace_instruction(s: str) -> str:
    old = "Let's think step by step and output the final answer after \"####\"."
    new = r"Let's think step by step and output the final answer within \boxed{}."
    if old not in s:
        raise ValueError('Pattern not found: expected exact phrase with "####" in quotes.')
    return s.replace(old, new)


def main():
    # ap = argparse.ArgumentParser(description="Convert rollout Parquet to ShareGPT JSONL (keep <think>).</think>")
    # ap.add_argument("--input", "-i", required=True, help="Path to input parquet file.")
    # ap.add_argument("--output", "-o", default=None, help="Path to output JSONL (default: alongside input).")
    # args = ap.parse_args()

    # in_path = Path(args.input)
    # if args.output is None:
    #     out_path = in_path.with_suffix(".sharegpt.jsonl")
    # else:
    #     out_path = Path(args.output)

    # out_path = Path("data/Llama3_GSM8K_rollout/sharegpt.jsonl")
    # in_path = Path("data/Llama3_GSM8K_rollout/result.parquet")
    out_path = Path("data/Llama3_GSM8K_run2_ckpt300_40rollouts/sharegpt.jsonl")
    in_path = Path("data/Llama3_GSM8K_run2_ckpt300_40rollouts/result.parquet")
    
    df = pd.read_parquet(in_path)
    n_rows = len(df)
    n_out = 0

    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            prompt = extract_prompt(row.get("prompt", ""))
            prompt = replace_instruction(prompt)
            responses = ensure_iterable_responses(row.get("responses", []))
            if not prompt or not responses:
                continue
            for r in responses:
                sample = build_sharegpt_sample(prompt, r)
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"Read rows: {n_rows}")
    print(f"Wrote ShareGPT samples: {n_out}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
