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
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-160",
    # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-320",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-480",
    # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-640",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-800",
    # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-960",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-1120",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-1280",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-160",
    # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-320",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-480",
    # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-640",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-800",
    # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-960",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-1120",
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-1280",
]


def read_mmlu_average(results_log_path: str) -> float | None:
    """从 results.log 中提取 'Average: xx.xx' 的数值（百分制）。"""
    if not os.path.exists(results_log_path):
        return None
    avg_re = re.compile(r"^\s*Average:\s*([0-9]+(?:\.[0-9]+)?)\s*$")
    with open(results_log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = avg_re.search(line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return None
    return None

records = []
missing = []

for path in MODEL_PATHS:
    results_log = os.path.join(path, "mmlu_eval", "results.log")
    avg = read_mmlu_average(results_log)
    if avg is None:
        missing.append(results_log)
        continue

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
    if "deepscaler" in path:  # RL: convert global_step to (#samples) by ×16
        step = step * 16

    records.append({"family": family, "path": path, "step": step, "mmlu_avg": avg})

# 汇总与打印缺失项（若有）
if missing:
    print("⚠️ 以下路径未找到或未解析到 Average：")
    for p in missing:
        print("   -", p)

res_df = pd.DataFrame(records)
print(res_df.sort_values(["family", "step"]))

# 画图：各 family 的曲线（Y 轴百分制）
plt.figure(figsize=(5, 5))
for fam, sub in res_df.groupby("family"):
    if fam == "Other":
        continue
    sub = sub.sort_values("step")
    
    steps = [0] + sub["step"].tolist()
    mmlu_avg = [53.49] + sub["mmlu_avg"].tolist()
    
        
    plt.plot(steps, mmlu_avg, marker="o", label=fam)
    
    print(fam)
    print(steps)
    print(mmlu_avg)

plt.xlabel("Training Step")
plt.ylabel("MMLU (Average, %)")
plt.title("Model MMLU (Average) vs Training Step")
plt.legend()
plt.grid(True)
plt.tight_layout()

fig_path = "experiments/10.19_finetune_Llama3_GSM8k/eval_models_16rollouts/figs/model_mmlu_average.pdf"
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, bbox_inches="tight")
print(f"✅ Saved to {fig_path}")