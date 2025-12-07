import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForImageClassification,
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    AutoImageProcessor,
)

# 复用现有模块
from config import Config
from model import (
    HumanLikeMultimodalModel,
    MultimodalModelDrop,
    MultiModalAttnCLSModel,
    MultiModalAttnModel,
)
from dataset import get_transforms
from mix_dataset import CausalConflictDataset

# 设置环境
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def run_causal_inference(model, dataloader, device):
    """
    运行推理并收集每个样本的预测概率和熵。
    """
    model.eval()
    results = []

    print("\n开始因果冲突推理...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            pixel_values = batch["pixel_values"].to(device)
            input_values = batch["input_values"].to(device)
            conditions = batch["condition"]  # list of strings

            outputs = model(pixel_values, input_values)
            logits = outputs["logits"]

            # 计算概率分布 (Softmax)
            probs = torch.softmax(logits, dim=-1)

            # 遍历 batch 中的每个样本
            for i in range(len(conditions)):
                # 获取 "Dog" (index 1) 的概率
                p_dog = probs[i, 1].item()

                # 计算熵 (Entropy) = -sum(p * log(p))
                # 熵越高表示模型越困惑/不确定
                entropy = -torch.sum(probs[i] * torch.log(probs[i] + 1e-9)).item()

                results.append(
                    {"Condition": conditions[i], "P(Dog)": p_dog, "Entropy": entropy}
                )

    return pd.DataFrame(results)


def plot_causal_results(df, save_dir):
    """
    绘制条形图展示不同条件下的决策倾向和不确定性。
    """
    # 设置绘图风格
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(16, 7))

    # 定义条件顺序，方便对比
    order = ["Congruent_Cat", "Congruent_Dog", "Conflict_V_Cat", "Conflict_V_Dog"]

    # --- 子图 1: 决策概率 P(Dog) ---
    plt.subplot(1, 2, 1)
    sns.barplot(
        data=df,
        x="Condition",
        y="P(Dog)",
        order=order,
        errorbar="sd",
        palette="viridis",
        capsize=0.1,
    )
    plt.title("Decision Probability (P(Dog))")
    plt.ylabel("Probability of choosing Dog")
    plt.ylim(0, 1.1)
    plt.axhline(0.5, color="r", linestyle="--", alpha=0.5, label="Chance Level")
    plt.xticks(rotation=25)

    # --- 子图 2: 不确定性 (Entropy) ---
    plt.subplot(1, 2, 2)
    sns.barplot(
        data=df,
        x="Condition",
        y="Entropy",
        order=order,
        errorbar="sd",
        palette="magma",
        capsize=0.1,
    )
    plt.title("Model Uncertainty (Entropy)")
    plt.ylabel("Entropy (bits)")
    plt.xticks(rotation=25)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "causal_conflict_analysis.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n图表已保存至: {save_path}")

    # --- 计算统计指标 ---
    print("\n========== 统计分析 ==========")
    mean_vals = df.groupby("Condition")[["P(Dog)", "Entropy"]].mean()
    print(mean_vals)

    # 计算视觉偏好指数 (Visual Bias Index)
    # Conflict_V_Dog (视狗听猫) -> 应该倾向于 1 (Dog)
    # Conflict_V_Cat (视猫听狗) -> 应该倾向于 0 (Cat)
    if "Conflict_V_Dog" in mean_vals.index and "Conflict_V_Cat" in mean_vals.index:
        p_v_dog = mean_vals.loc["Conflict_V_Dog", "P(Dog)"]
        p_v_cat = mean_vals.loc["Conflict_V_Cat", "P(Dog)"]

        # Bias = (P(D|V_D) - P(D|V_C) + 1) / 2
        # 1.0 = 完全视觉主导, 0.0 = 完全听觉主导
        visual_bias = (p_v_dog - p_v_cat + 1) / 2

        print("-" * 30)
        print(f"Visual Bias Index: {visual_bias:.4f}")
        print("(1.0 = Visual Dominance, 0.0 = Audio Dominance, 0.5 = No Bias)")

        # 保存统计结果到 JSON
        stats_path = os.path.join(save_dir, "causal_stats.json")
        stats_dict = mean_vals.to_dict()
        stats_dict["visual_bias"] = visual_bias
        with open(stats_path, "w") as f:
            json.dump(stats_dict, f, indent=4)


def main():
    # 1. 配置路径 (请根据需要修改 json_path)
    json_path = "/home/amax/dakai/neuron/checkpoints/Drop_lr1e-04_bs16_mask0.7_v0.5_a0.5_1204_0954/config.json"

    # 纯净数据路径 (用于构建冲突样本)
    img_cache_dir = "/home/amax/dakai/dataset/microsoft"
    audio_data_dir = "/home/amax/dakai/dataset/dc_w/DvC"

    # 2. 加载配置
    if not os.path.exists(json_path):
        print(f"错误: 找不到配置文件 {json_path}")
        return

    cfg = Config.read_json(json_path, eval_mode=True)
    device = cfg.device
    print(f"使用设备: {device}")
    print(f"模型类型: {cfg.model_type}")

    # 3. 准备数据处理工具
    # 图像
    _, val_transforms = get_transforms()

    # 音频
    audio_model_cache = "/home/amax/dakai/neuron/model/facebook/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        audio_model_cache, use_fast=True
    )

    # 4. 初始化因果冲突数据集
    causal_ds = CausalConflictDataset(
        image_data_dir=img_cache_dir,
        audio_data_dir=audio_data_dir,
        image_transforms=val_transforms,
        audio_feature_extractor=feature_extractor,
        samples_per_condition=200,  # 每种条件测试 200 个样本
        seed=cfg.seed,
    )

    dataloader = DataLoader(causal_ds, batch_size=32, shuffle=False, num_workers=4)

    # 5. 加载模型
    print("正在加载模型...")
    img_model_best = AutoModelForImageClassification.from_pretrained(cfg.img_model_path)
    audio_model_best = AutoModelForAudioClassification.from_pretrained(
        cfg.audio_best_path
    )
    img_model_best.to(device)
    audio_model_best.to(device)
    mp = cfg.model_params

    match cfg.model_type:
        case "HumanLikeMultimodalModel":
            model = HumanLikeMultimodalModel(
                img_model_best,
                audio_model_best,
                shared_dim=mp["shared_dim"],
                num_classes=mp["num_classes"],
            ).to(device)
        case "MultimodalModelDrop":
            model = MultimodalModelDrop(
                img_model_best,
                audio_model_best,
                shared_dim=mp["shared_dim"],
                num_classes=mp["num_classes"],
                vision_drop_prob=mp["vision_drop_prob"],
                audio_drop_prob=mp["audio_drop_prob"],
                emb_mask_prob=mp["emb_mask_prob"],
            ).to(device)
        case "MultiModalAttnModel":
            model = MultiModalAttnModel(
                img_model_best,
                audio_model_best,
                shared_dim=mp["shared_dim"],
                num_classes=mp["num_classes"],
                attn_heads=mp["attn_heads"],
                attn_dropout=mp["attn_dropout"],
                vision_drop_prob=mp["vision_drop_prob"],
                audio_drop_prob=mp["audio_drop_prob"],
            ).to(device)
        case "MultiModalAttnCLSModel":
            model = MultiModalAttnCLSModel(
                img_model_best,
                audio_model_best,
                shared_dim=mp["shared_dim"],
                num_classes=mp["num_classes"],
                attn_heads=mp["attn_heads"],
                attn_dropout=mp["attn_dropout"],
                vision_drop_prob=mp["vision_drop_prob"],
                audio_drop_prob=mp["audio_drop_prob"],
            ).to(device)
        case _:
            raise ValueError(f"未知的多模态模型类型: {cfg.model_type}")

    model = model.to(device)

    save_path = cfg.save_path
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print("模型加载成功！")

    # 6. 运行实验
    df_results = run_causal_inference(model, dataloader, device)

    # 7. 绘图与保存
    plot_causal_results(df_results, cfg.ckpt_dir)


if __name__ == "__main__":
    main()
