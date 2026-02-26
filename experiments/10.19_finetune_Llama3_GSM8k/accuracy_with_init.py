import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATHS = [
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/finetune_Llama3B/checkpoint-3054",
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/deepscaler-1.5b-8k_run2/actor/global_step_20",
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/deepscaler-1.5b-8k_run2/actor/global_step_40",
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/deepscaler-1.5b-8k_run2/actor/global_step_60",
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/deepscaler-1.5b-8k_run2/actor/global_step_80",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk/checkpoint-160",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk/checkpoint-320",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk/checkpoint-480",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk/checkpoint-640",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk/checkpoint-800",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk/checkpoint-960",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk/checkpoint-1120",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk/checkpoint-1280",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained/checkpoint-160",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained/checkpoint-320",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained/checkpoint-480",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained/checkpoint-640",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained/checkpoint-800",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained/checkpoint-960",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained/checkpoint-1120",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained/checkpoint-1280",
]

def extract_boxed_answer(text: str):
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    return m.group(1).strip() if m else None

records = []

for path in MODEL_PATHS:
    file_path = os.path.join(path, "boxed_test_result__shards", "batch_000000.parquet")
    if not os.path.exists(file_path):
        print(f"[Skip] {file_path} not found")
        continue

    df = pd.read_parquet(file_path)
    total, correct = 0, 0

    for _, row in df.iterrows():
        gt = str(row["reward_model"].get("ground_truth", "")).strip()
        responses = row["responses"]
        if isinstance(responses, np.ndarray):
            responses = responses[0]
        pred = extract_boxed_answer(str(responses))
        if pred is not None:
            total += 1
            if pred == gt:
                correct += 1

    acc = correct / total if total > 0 else np.nan

    # determine family and step
    if "deepscaler" in path:
        family = "GRPO"
    elif "SFT_dpsk" in path:
        family = "SFT (DPSK)"
    elif "SFT_rltrained" in path:
        family = "SFT (RL-Trained)"
    else:
        family = "Other"
    step = int(re.findall(r"(?:global_step_|checkpoint-)(\d+)", path)[0])
    if "deepscaler" in path: # RL 
        step = step * 16 
    records.append({"family": family, "path": path, "step": step, "accuracy": acc, "total": total})

res_df = pd.DataFrame(records)
print(res_df)

# plot
plt.figure(figsize=(5,5))
for fam, sub in res_df.groupby("family"):
    if fam == "Other":
        continue
    sub = sub.sort_values("step")
    steps = [0] + sub["step"].tolist()
    accuracies = [0.643389] + sub["accuracy"].tolist()
    plt.plot(steps, accuracies, marker="o", label=fam)

plt.xlabel("Training Step")
plt.ylabel("Accuracy")
plt.title("Model Accuracy on GSM8K Test")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("experiments/10.19_finetune_Llama3_GSM8k/model_accuracy_comparison.pdf")
print("✅ Saved to model_accuracy_comparison.pdf")
