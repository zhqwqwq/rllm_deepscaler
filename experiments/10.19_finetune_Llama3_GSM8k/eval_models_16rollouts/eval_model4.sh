set -euo pipefail


MODEL_PATHS=(
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/deepscaler-1.5b-8k_run2/actor/global_step_20"
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/deepscaler-1.5b-8k_run2/actor/global_step_40"
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/deepscaler-1.5b-8k_run2/actor/global_step_60"
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/deepscaler-1.5b-8k_run2/actor/global_step_80"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-160"
    # # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-320"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-480"
    # # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-640"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-800"
    # # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-960"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-1120"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-1280"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-160"
    # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-320"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-480"
    # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-640"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-800"
    # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-960"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-1120"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-1280"
)

DATA_PATH="/cephfs/zhanghuaqing/RL/rllm_deepscaler/data/gsm8k_boxed/test.parquet"

# 温度顺序：0, 0.7, 0.8, 0.6
# TEMPS=("0" "0.7" "0.8" "0.6")

TEMPS=("0.8" "0.6")

for TEMP in "${TEMPS[@]}"; do
  for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    if [[ "$TEMP" == "0.7" ]]; then
      N_SAMPLES=8
    else
      N_SAMPLES=1
    fi

    # 文件名里用 p 代替小数点，例：0.7 -> 0p7
    TEMP_TAG="${TEMP/./p}"
    OUTPUT_PATH="${MODEL_PATH}/boxed_test_result_T${TEMP_TAG}_S${N_SAMPLES}.parquet"

    echo "Model Path: ${MODEL_PATH}"
    echo "Dataset:    ${DATA_PATH}"
    echo "Temp:       ${TEMP}  (samples=${N_SAMPLES})"
    echo "Output:     ${OUTPUT_PATH}"

    # 确保目录存在
    mkdir -p "$(dirname "$OUTPUT_PATH")"

    python3 -m verl.trainer.robust_generation_from_TinyZero \
      trainer.nnodes=1 \
      trainer.n_gpus_per_node=8 \
      data.path="${DATA_PATH}" \
      data.output_path="${OUTPUT_PATH}" \
      data.n_samples="${N_SAMPLES}" \
      data.batch_size=2048 \
      model.path="${MODEL_PATH}" \
      rollout.temperature="${TEMP}" \
      rollout.response_length=4096 \
      rollout.top_k=50 \
      rollout.top_p=0.95 \
      rollout.gpu_memory_utilization=0.9 \
      rollout.tensor_model_parallel_size=1
  done
done