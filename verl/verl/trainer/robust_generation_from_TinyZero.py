# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""
Generate responses given a dataset of prompts (robust, per-batch saving)
"""
import os
import ray
import numpy as np
import hydra
import pandas as pd
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from pprint import pprint

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker

import logging
import traceback

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'


def atomic_write_parquet(df: pd.DataFrame, path: str):
    """Write df to path atomically: write to tmp then rename."""
    tmp = path + ".tmp"
    df.to_parquet(tmp)
    os.replace(tmp, path)  # atomic on POSIX


def shard_path_from_output(base_output_path: str, batch_idx: int) -> str:
    """
    Make per-batch shard path based on the configured output_path.
    If output_path = /foo/bar/out.parquet, shards will be /foo/bar/out__shards/batch_000123.parquet
    """
    base_dir = os.path.dirname(base_output_path)
    base_name = os.path.splitext(os.path.basename(base_output_path))[0]
    shard_dir = os.path.join(base_dir, f"{base_name}__shards")
    makedirs(shard_dir, exist_ok=True)
    return os.path.join(shard_dir, f"batch_{batch_idx:06d}.parquet")


def already_done(shard_path: str) -> bool:
    return os.path.exists(shard_path)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _generate_once_with_retry(wg: RayWorkerGroup, data: DataProto):
    """
    A single generate call with retry. Exceptions will be retried by tenacity.
    """
    return wg.generate_sequences(data)


# def try_generate_n_samples(wg: RayWorkerGroup, data: DataProto, n_samples: int, real_batch_size: int):
#     """
#     Robust wrapper that attempts to generate n_samples lists of decoded strings for a single batch.
#     If a given sample round fails after retries, we record a placeholder "" to keep shape consistent.
#     """
#     outputs_all = [[] for _ in range(n_samples)]
#     for i in range(n_samples):
#         try:
#             print(f"  - Sample {i+1}/{n_samples} generating ...")
#             out = _generate_once_with_retry(wg, data)  # may raise -> auto-retry
#             out = out[:real_batch_size]  # strip dummy paddings
#             outputs_all[i].append(out)
#         except Exception as e:
#             print(f"  ! Sample {i+1}/{n_samples} failed after retries: {repr(e)}")
#             # create an empty placeholder DataProto with same batch size to keep shapes aligned
#             outputs_all[i].append(None)
#     return outputs_all  # list length n_samples; each item is DataProto or None

def try_generate_n_samples(wg: RayWorkerGroup, data: DataProto, n_samples: int, real_batch_size: int):
    """
    Robust wrapper that attempts to generate n_samples lists of decoded strings for a single batch.
    If a given sample round fails after retries, log the error, restart worker, and continue.
    """
    outputs_all = [[] for _ in range(n_samples)]
    for i in range(n_samples):
        try:
            print(f"  - Sample {i+1}/{n_samples} generating ...")
            out = _generate_once_with_retry(wg, data)  # may raise -> auto-retry
            out = out[:real_batch_size]  # strip dummy paddings
            outputs_all[i].append(out)
        except Exception as e:
            # 1. 写详细 log（包含 traceback）
            err_msg = f"Sample {i+1}/{n_samples} failed after retries: {repr(e)}\n{traceback.format_exc()}"
            logging.error(err_msg)

            # 2. 重启 worker，防止 GPU context 污染
            try:
                print(f"  ! Restarting workers after failure at sample {i+1} ...")
                wg.restart_workers()
                wg.init_model()
            except Exception as restart_e:
                logging.error(f"Failed to restart workers: {repr(restart_e)}")

            # 3. 保持 shape 一致，填 None 占位
            outputs_all[i].append(None)
    return outputs_all


def decode_and_unpad(tokenizer: AutoTokenizer, seq_batch, response_length: int):
    """
    seq_batch: DataProto or None
    Return: list[str] of length real_batch_size
    """
    if seq_batch is None:
        return None  # caller will handle and insert placeholders
    text = tokenizer.batch_decode(
        seq_batch.batch['input_ids'][:, -response_length:],
        skip_special_tokens=False
    )
    pad_token = tokenizer.pad_token
    if pad_token is None:
        return text
    return [t.replace(pad_token, '') for t in text]


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    print("!!!!")
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    print("????")
    
    
    # === 新增: 设置 log 文件路径到 output_path 同目录 ===
    log_dir = os.path.dirname(config.data.output_path)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "generation_errors.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging errors to {log_path}")

    local_path = copy_local_path_from_hdfs(config.model.path)
    print("local_path", local_path)
    print("output path", config.data.output_path)

    # tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # dataset
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [c.tolist() for c in chat_lst]  # each row already chat-template list[dict]

    # ray workers (rollout)
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
    num_batch = (total_samples + config_batch_size - 1) // config_batch_size

    print(f"Total {total_samples} rows, batch_size={config_batch_size}, num_batch={num_batch}, dp_size={dp_size}")

    # start_idx = config.rollout.start_idx if config.rollout.start_idx != None else 0
    # end_idx = config.rollout.end_idx if config.rollout.end_idx != None else num_batch
    # # for batch_idx in range(num_batch):
    # print(f"Processing batches from {start_idx} to {end_idx} ...")
    for batch_idx in range(num_batch):
        print(batch_idx)
        start = batch_idx * config_batch_size
        end = min((batch_idx + 1) * config_batch_size, total_samples)
        shard_path = shard_path_from_output(config.data.output_path, batch_idx)

        # 支持断点续跑：若 shard 已存在，直接跳过
        if already_done(shard_path):
            print(f"[{batch_idx+1}/{num_batch}] shard exists, skip -> {shard_path}")
            continue

        print(f"[{batch_idx+1}/{num_batch}] Start to process rows [{start}:{end}) ...")
        batch_chat_lst = chat_lst[start:end]

        # tokenize
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors='pt',
            return_dict=True,
            tokenize=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
        data = DataProto.from_dict(batch_dict)
        real_batch_size = data.batch['input_ids'].shape[0]

        # dp padding to be divisible
        if real_batch_size % dp_size != 0:
            dummy_size = dp_size - (real_batch_size % dp_size)
            dummy_data = data[:dummy_size]
            data = DataProto.concat([data, dummy_data])
            print(f"  dp_size {dp_size} not divisible by real_batch_size {real_batch_size}, add {dummy_size} dummy data")
        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dp_size == 0, f"batch_size {batch_size} is not divisible by dp_size {dp_size}"

        # generate with retries, per-sample
        print(f"[{batch_idx+1}/{num_batch}] Start to generate n_samples={config.data.n_samples} ...")
        seq_outputs_all = try_generate_n_samples(wg, data, config.data.n_samples, real_batch_size)

        # decode & unpad; shape -> (n_samples, real_batch_size) -> transpose to (real_batch_size, n_samples)
        decoded_all = []
        for seq_dp in seq_outputs_all:
            # seq_dp is a list with a single element DataProto/None (we kept list form to be extendable)
            dp0 = seq_dp[0]
            decoded_all.append(decode_and_unpad(tokenizer, dp0, config.rollout.response_length))

        # transpose to per-row
        # if any decode failed (None), we fill placeholders "" to keep same length
        per_row_responses = []
        for row_i in range(real_batch_size):
            row_list = []
            for s_i in range(config.data.n_samples):
                if decoded_all[s_i] is None:
                    row_list.append("")  # placeholder when that sample failed
                else:
                    row_list.append(decoded_all[s_i][row_i])
            per_row_responses.append(row_list)

        # build shard dataframe: take the slice of original dataset and add 'responses'
        shard_df = dataset.iloc[start:end].copy()
        shard_df['responses'] = per_row_responses

        # save atomically for this batch
        try:
            atomic_write_parquet(shard_df, shard_path)
            print(f"[{batch_idx+1}/{num_batch}] Saved shard -> {shard_path}")
        except Exception as e:
            print(f"[{batch_idx+1}/{num_batch}] ERROR saving shard: {repr(e)}")
            # 不抛出，继续跑后续 batch；也可选择 raise 终止
            # raise

    print("All batches processed. Shards are stored under: ",
          os.path.join(os.path.dirname(config.data.output_path),
                       os.path.splitext(os.path.basename(config.data.output_path))[0] + "__shards"))

    # 可选：如需最终合并成一个大 parquet（可能很大），可在单独脚本中合并，
    # 避免途中失败导致单文件损坏。这里不做强制合并。
    return None


if __name__ == '__main__':
    main()
