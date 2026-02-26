set -x

MODEL_PATH="/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/Llama3_GRPO_run2/actor/global_step_300" # actually is GSM8K
DATA_PATH="/cephfs/zhanghuaqing/RL/rllm_deepscaler/data/gsm8k/train.parquet"
OUTPUT_DIR="/cephfs/zhanghuaqing/RL/rllm_deepscaler/data/Llama3_GSM8K_run2_ckpt300_40rollouts"

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATA_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"

# Loop through all datatypes
python3 -m verl.trainer.robust_generation_from_TinyZero \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=${DATA_PATH} \
    data.output_path=${OUTPUT_DIR}/result.parquet \
    data.n_samples=47 \
    data.batch_size=2048 \
    model.path=${MODEL_PATH} \
    rollout.temperature=1 \
    rollout.response_length=4096 \
    rollout.top_k=-1 \
    rollout.top_p=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.tensor_model_parallel_size=1
