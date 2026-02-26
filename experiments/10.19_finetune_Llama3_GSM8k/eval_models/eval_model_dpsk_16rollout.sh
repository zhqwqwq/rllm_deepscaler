set -x


MODEL_PATHS=(
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-160"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-320"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-480"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-640"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-800"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-960"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-1120"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-1280"
)

DATA_PATH="/cephfs/zhanghuaqing/RL/rllm_deepscaler/data/gsm8k_boxed/test.parquet"

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    OUTPUT_PATH=${MODEL_PATH}/boxed_test_result.parquet

    # Echo the values for verification
    echo "Model Path: ${MODEL_PATH}"
    echo "Datasets: ${DATA_PATH}"
    echo "Output Path: ${OUTPUT_PATH}"

    # Loop through all datatypes
    python3 -m verl.trainer.robust_generation_from_TinyZero \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=4 \
        data.path=${DATA_PATH} \
        data.output_path=${OUTPUT_PATH} \
        data.n_samples=1 \
        data.batch_size=2048 \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.6 \
        rollout.response_length=4096 \
        rollout.top_k=50 \
        rollout.top_p=0.95 \
        rollout.gpu_memory_utilization=0.9 \
        rollout.tensor_model_parallel_size=1
done