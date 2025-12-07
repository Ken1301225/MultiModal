import json
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoImageProcessor


# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"

from model import (
    HumanLikeMultimodalModel,
    MultimodalModelDrop,
    MultiModalAttnCLSModel,
    MultiModalAttnModel,
)


def inference(
    inference_model,
    image_model_best,
    audio_model_best,
    mix_dataset,
    device="cuda",
    mode=["mix"],
    eval_mode=["choice_proportion"],
):
    """
    对 PairedDatasetFactory 生成的每个 (pair_sample, seq_id) 计算
    mix / image / audio 三种方式的准确率，并在命令行打印表格。
    """
    # 支持传入字符串
    if isinstance(mode, str):
        mode = [mode]

    print("\n开始在配对测试集上进行评估...")
    progress_bar = tqdm(mix_dataset, desc="Testing")

    # 保存所有 seq_id 的结果
    acc_rows = []  # 每个元素: [seq_id, num_samples, acc_mix, acc_img, acc_audio]
    p_rows = []  # 每个元素: [seq_id, num_samples, acc_mix, acc_img, acc_audio]

    for pair_sample, seq_id in progress_bar:
        total_samples = len(pair_sample)
        if total_samples == 0:
            continue
        mix_pred_1_cnt = 0
        img_pred_1_cnt = 0
        audio_pred_1_cnt = 0
        ground_truths = []
        mix_preds, img_preds, audio_preds = [], [], []
        logits_v_accum = []
        logits_a_accum = []

        for item in pair_sample:
            image = item["pixel_values"]
            audio = item["input_values"]
            true_label_id = item["labels"]
            if isinstance(true_label_id, torch.Tensor):
                true_label_id = true_label_id.item()
            ground_truths.append(true_label_id)

            # 移动到设备
            pixel_values = image.unsqueeze(0).to(device)
            input_values = audio.unsqueeze(0).to(device)
            img_inputs = {"pixel_values": pixel_values}
            audio_inputs = {"input_values": input_values}

            with torch.no_grad():
                if "mix" in mode:
                    mix_outputs = inference_model(pixel_values, input_values)
                    mix_logits = mix_outputs["logits"]
                    mix_pred_idx = torch.argmax(mix_logits, dim=-1).item()
                    if mix_pred_idx == 1:
                        mix_pred_1_cnt += 1
                    mix_preds.append(mix_pred_idx)

                if "image" in mode:
                    img_outputs = image_model_best(**img_inputs)
                    img_pred_idx = img_outputs.logits.argmax(-1).item()
                    img_preds.append(img_pred_idx)
                    if img_pred_idx == 1:
                        img_pred_1_cnt += 1
                    logits_v_accum.append(img_outputs.logits)

                if "audio" in mode:
                    audio_outputs = audio_model_best(**audio_inputs)
                    audio_pred_idx = audio_outputs.logits.argmax(-1).item()
                    audio_preds.append(audio_pred_idx)
                    if audio_pred_idx == 1:
                        audio_pred_1_cnt += 1
                    logits_a_accum.append(audio_outputs.logits)
        all_logits_v = torch.cat(logits_v_accum)
        all_logits_a = torch.cat(logits_a_accum)
        
        rel_logits_v = all_logits_v[:, 1] - all_logits_v[:, 0]
        rel_logits_a = all_logits_a[:, 1] - all_logits_a[:, 0]

        if calib_params:
            a_v, b_v = calib_params["img"]
            a_a, b_a = calib_params["audio"]

            calib_v = (rel_logits_v * a_v) + b_v
            calib_a = (rel_logits_a * a_a) + b_a

            final_score = calib_v + calib_a
        else:
            final_score = rel_logits_v + rel_logits_a

        p_baseline_1 = (final_score > 0).float().mean().item()

        # --- 关键：将当前 Pair 的结果存入列表 ---

        # 计算该 seq_id 下三种方式的准确率
        acc_mix = (
            accuracy_score(ground_truths, mix_preds)
            if ("mix" in mode and len(mix_preds) > 0)
            else None
        )
        acc_img = (
            accuracy_score(ground_truths, img_preds)
            if ("image" in mode and len(img_preds) > 0)
            else None
        )
        acc_audio = (
            accuracy_score(ground_truths, audio_preds)
            if ("audio" in mode and len(audio_preds) > 0)
            else None
        )
        p_mix_1 = (
            mix_pred_1_cnt / total_samples
            if ("mix" in mode and total_samples > 0)
            else None
        )
        p_img_1 = (
            img_pred_1_cnt / total_samples
            if ("image" in mode and total_samples > 0)
            else None
        )
        p_audio_1 = (
            audio_pred_1_cnt / total_samples
            if ("audio" in mode and total_samples > 0)
            else None
        )

        p_rows.append(
            [seq_id, total_samples, p_mix_1, p_img_1, p_audio_1, p_baseline_1]
        )
        acc_rows.append([seq_id, total_samples, acc_mix, acc_img, acc_audio])

    if "acc" in eval_mode:
        # -------- 在命令行打印表格 --------
        if not acc_rows:
            print("没有任何评估结果。")
            return

        # 先按 seq_id 排序
        acc_rows.sort(key=lambda x: x[0])

        # 表头
        header = ["seq_id", "num_samples", "acc_mix", "acc_image", "acc_audio"]
        col_widths = [10, 12, 10, 10, 10]

        def fmt_cell(val, width):
            if val is None:
                s = "-"
            elif isinstance(val, float):
                s = f"{val:.4f}"
            else:
                s = str(val)
            return s.ljust(width)

        print("\n========== 各 seq_id 下不同方法的准确率表 ==========")
        # 打印表头
        header_line = " | ".join(fmt_cell(h, w) for h, w in zip(header, col_widths))
        print(header_line)
        print("-" * len(header_line))

        # 打印每一行
        for seq_id, num_samples, acc_mix, acc_img, acc_audio in acc_rows:
            line = " | ".join(
                [
                    fmt_cell(seq_id, col_widths[0]),
                    fmt_cell(num_samples, col_widths[1]),
                    fmt_cell(acc_mix, col_widths[2]),
                    fmt_cell(acc_img, col_widths[3]),
                    fmt_cell(acc_audio, col_widths[4]),
                ]
            )
            print(line)

        print("=================================================\n")
    if "choice_proportion" in eval_mode:
        return p_rows


def plot_sigmoid(x, y, label, color):
    """
    使用 Logistic（Sigmoid）函数拟合曲线，同时画出原始点。
    Logistic: y = 1 / (1 + exp(-k*(x - x0)))
    其中 x0 是拐点（PSE），k 是斜率（陡峭程度）。
    """
    from scipy.optimize import curve_fit

    xs, ys = [], []
    for xi, yi in zip(x, y):
        if yi is not None:
            xs.append(float(xi))
            ys.append(float(yi))

    if len(xs) < 4:
        # 点太少，无法拟合，直接画散点
        plt.plot(xs, ys, "o", label=label, color=color)
        return

    xs = np.array(xs)
    ys = np.array(ys)

    # 按 x 排序
    sorted_idx = np.argsort(xs)
    xs = xs[sorted_idx]
    ys = ys[sorted_idx]

    # Logistic 函数定义
    def logistic(x, x0, k):
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))

    try:
        # 拟合参数：x0 初值取中点，k 初值取 10
        popt, _ = curve_fit(
            logistic,
            xs,
            ys,
            p0=[0.5, 10.0],
            bounds=([0.0, 0.1], [1.0, 100.0]),  # x0 在 [0,1]，k 正数
            maxfev=5000,
        )
        x0_fit, k_fit = popt

        # 生成平滑曲线
        x_dense = np.linspace(xs.min(), xs.max(), 300)
        y_dense = logistic(x_dense, x0_fit, k_fit)

        # 原始点（淡一些）
        plt.plot(xs, ys, "o", color=color, alpha=0.4, markersize=5)
        # 拟合曲线
        plt.plot(
            x_dense,
            y_dense,
            "-",
            color=color,
            label=f"{label} (x0={x0_fit:.2f}, k={k_fit:.1f})",
        )

        print(f"[{label}] Logistic 拟合: x0(PSE)={x0_fit:.4f}, k(slope)={k_fit:.4f}")
        return x0_fit, k_fit

    except Exception as e:
        print(f"Logistic 拟合失败（{label}）：{e}，退回散点。")
        plt.plot(xs, ys, "o", label=label, color=color)
        

def plot_logit_distribution(
    inference_model,
    image_model,
    audio_model,
    dataset,
    device,
    class_name="dog",
    label_id=1,
):
    """
    对特定类别的数据集，收集并绘制单模态和多模态模型的 Logit 分布。

    Args:
        inference_model: 混合模型。
        image_model: 纯图像模型。
        audio_model: 纯音频模型。
        dataset: 只包含特定类别样本的数据集。
        device: 'cuda' or 'cpu'。
        class_name: 类别名称，用于绘图。
        label_id: 类别对应的标签 ID。
    """
    import seaborn as sns
    from scipy.stats import norm

    print(f"\n开始为 '{class_name}' 类别收集 Logits...")

    mix_logits_list, img_logits_list, audio_logits_list = [], [], []

    progress_bar = tqdm(dataset, desc=f"Collecting logits for {class_name}")
    for item in progress_bar:
        image = item["pixel_values"].unsqueeze(0).to(device)
        audio = item["input_values"].unsqueeze(0).to(device)

        with torch.no_grad():
            # 图像模型
            img_outputs = image_model(pixel_values=image)
            img_logits = img_outputs.logits[0]
            # Logit for the target class (dog 1 cat 0)
            img_logits_list.append(img_logits[label_id].item())

            # 音频模型
            audio_outputs = audio_model(input_values=audio)
            audio_logits = audio_outputs.logits[0]
            audio_logits_list.append(audio_logits[label_id].item())

            # 混合模型
            mix_outputs = inference_model(image, audio)
            mix_logits = mix_outputs["logits"][0]
            mix_logits_list.append(mix_logits[label_id].item())

    # 转换为 NumPy 数组
    mix_logits_arr = np.array(mix_logits_list)
    img_logits_arr = np.array(img_logits_list)
    audio_logits_arr = np.array(audio_logits_list)
    
    # 1. 提取统计值 (使用之前计算好的 stats_results 或重新计算)
    mu_V, std_V = img_logits_arr.mean(), img_logits_arr.std()
    mu_A, std_A = audio_logits_arr.mean(), audio_logits_arr.std()
    mu_Mix, std_Mix = mix_logits_arr.mean(), mix_logits_arr.std() # 混合模型通常不需要标准化，但为了公平比较可以处理

    # 2. 对所有模态 Logits 进行 Z-score 标准化
    # **注意：只对单模态进行标准化，或者对 Mix 也做，取决于你的比较目的。**
    # 目标是比较**模态贡献**，因此只标准化 Visual 和 Audio 是更科学的选择。

    # 标准化 Visual Logits (Z-score)
    img_logits_norm = (img_logits_arr - mu_V) / std_V

    # 标准化 Audio Logits (Z-score)
    audio_logits_norm = (audio_logits_arr - mu_A) / std_A

    # Mix Logits 通常不归一化，因为它是最终结果，但我们可以进行 Z-score 归一化以观察其在相对空间中的集中度。
    mix_logits_norm = (mix_logits_arr - mu_Mix) / std_Mix

    # --- 绘图 ---
    plt.figure(figsize=(12, 7))

    # 使用 seaborn 绘制核密度估计图 (KDE Plot)
    sns.kdeplot(img_logits_norm, fill=True, color="tab:orange", label="Visual")
    sns.kdeplot(audio_logits_norm, fill=True, color="tab:green", label="Audio")
    sns.kdeplot(mix_logits_norm, fill=True, color="tab:blue", label="Mix")

    # --- 计算并打印统计数据 ---
    print("\n========== Logit 分布统计 ==========")
    models_data = {
        "Visual": img_logits_norm,
        "Audio": audio_logits_norm,
        "Mix": mix_logits_norm,
    }
    stats_results = {}
    for name, data in models_data.items():
        mean, std = norm.fit(data)
        stats_results[name] = {"mean": mean, "std": std}
        print(f"模型: {name}")
        print(f"  - 均值 (Mean): {mean:.4f}")
        print(f"  - 标准差 (Std Dev): {std:.4f}")
        # 在图上用虚线标出均值位置
        plt.axvline(mean, linestyle="--", color=sns.color_palette()[list(models_data.keys()).index(name)], alpha=0.6)

    plt.title(f"Logit Distribution for '{class_name}' Class")
    plt.xlabel(f"Logit value for '{class_name}'")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    return plt, stats_results


def set_seed(seed: int = 42):
    """
    设置 Python / NumPy / PyTorch 的随机种子，尽量保证可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果用到 CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
def get_global_z_score_params(model, dataset_factory, device, modality="image"):
    """
    遍历整个数据集工厂（所有混合度），计算全局的均值和标准差。
    已修复：添加 pad_collate 处理变长音频数据。
    """
    print(f"正在计算 {modality} 模态的【全局】Z-Score 参数...")

    model.eval()
    all_logits_list = []
    total_count = 0

    def pad_collate(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        input_values = [item["input_values"] for item in batch]
        labels = [item["labels"] for item in batch]

        pixel_values = torch.stack(pixel_values)

        if input_values[0].dim() > 1:
            input_values = [iv.squeeze() for iv in input_values]

        from torch.nn.utils.rnn import pad_sequence

        # batch_first=True 会生成 [Batch, Max_Len]
        input_values_padded = pad_sequence(
            input_values, batch_first=True, padding_value=0.0
        )

        labels = torch.tensor(labels)

        return {
            "pixel_values": pixel_values,
            "input_values": input_values_padded,
            "labels": labels,
        }

    with torch.no_grad():
        for dataset, seq_id in dataset_factory:

            # 使用自定义 collate_fn 创建 Loader
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=64,
                shuffle=False,
                num_workers=0,
                collate_fn=pad_collate,
            )

            for batch in loader:
                if modality == "image":
                    inputs = batch["pixel_values"].to(device)
                    outputs = model(pixel_values=inputs)
                elif modality == "audio":
                    inputs = batch["input_values"].to(device)
                    outputs = model(input_values=inputs)

                relative_logit = outputs.logits[:, 1] - outputs.logits[:, 0]
                all_logits_list.append(relative_logit.cpu())

            total_count += len(dataset)

    all_logits_tensor = torch.cat(all_logits_list, dim=0)

    mean_val = torch.mean(all_logits_tensor).item()
    std_val = torch.std(all_logits_tensor).item()

    if std_val < 1e-6:
        std_val = 1.0

    print(
        f"--> {modality} Global Params (N={total_count}) | Mean: {mean_val:.4f} | Std: {std_val:.4f}"
    )

    return 1.0 / std_val, -mean_val / std_val


if __name__ == "__main__":
    from transformers import (
        AutoModelForAudioClassification,
        AutoModelForImageClassification,
        AutoFeatureExtractor,
    )
    from mix_dataset import PairedDatasetFactory
    from dataset import (
        get_transforms,
    )
    import matplotlib.pyplot as plt
    from config import Config

    cfg = Config.read_json(
        json_path="E:/NeuralScience/project/MultiModal/checkpoints/Drop_lr1e-04_bs16_mask0.7_v0.5_a0.5_1204_0954/config.json",
        eval_mode=True,
    )

    set_seed(cfg.seed)

    img_model_cache = "E:/NeuralScience/project/MultiModal/cache_dir/restnet50/models--microsoft--resnet-50/snapshots/34c2154c194f829b11125337b98c8f5f9965ff19"
    image_processor = AutoImageProcessor.from_pretrained(img_model_cache, use_fast=True)

    audio_model_cache = "E:/NeuralScience/project/MultiModal/cache_dir/wav2vec2-base/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        audio_model_cache, use_fast=True
    )

    audio_data_dir = "E:/NeuralScience/project/mixed_cats_dogs_audio"
    img_data_dir = "E:/NeuralScience/project/mixed_cats_dogs_img"

    _train_transforms, _val_transforms = get_transforms()

    multi__ds = PairedDatasetFactory(
        img_data_dir, audio_data_dir, _train_transforms, feature_extractor
    )

    device = cfg.device
    img_model_best = AutoModelForImageClassification.from_pretrained(cfg.img_model_path)
    audio_model_best = AutoModelForAudioClassification.from_pretrained(
        cfg.audio_best_path
    )
    save_path = cfg.save_path

    img_model_best.to(device)
    audio_model_best.to(device)

    mp = cfg.model_params

    match cfg.model_type:
        case "HumanLikeMultimodalModel":
            inference_model = HumanLikeMultimodalModel(
                img_model_best,
                audio_model_best,
                shared_dim=mp["shared_dim"],
                num_classes=mp["num_classes"],
            ).to(device)
        case "MultimodalModelDrop":
            inference_model = MultimodalModelDrop(
                img_model_best,
                audio_model_best,
                shared_dim=mp["shared_dim"],
                num_classes=mp["num_classes"],
                vision_drop_prob=mp["vision_drop_prob"],
                audio_drop_prob=mp["audio_drop_prob"],
                emb_mask_prob=mp["emb_mask_prob"],
            ).to(device)
        case "MultiModalAttnModel":
            inference_model = MultiModalAttnModel(
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
            inference_model = MultiModalAttnCLSModel(
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

    inference_model.load_state_dict(torch.load(save_path, map_location=device))
    inference_model.eval()
    print("模型加载成功！")
    
    a_v, b_v = get_global_z_score_params(img_model_best, multi__ds, device, "image")
    a_a, b_a = get_global_z_score_params(audio_model_best, multi__ds, device, "audio")

    calib_params = {"img": (a_v, b_v), "audio": (a_a, b_a)}

    p_rows = inference(
        inference_model,
        img_model_best,
        audio_model_best,
        multi__ds,
        device,
        mode=["mix", "image", "audio"],
        eval_mode=["choice_proportion"],
        calib_params=calib_params,
    )
    p_rows.sort(key=lambda x: x[0])

    seq_ids = [row[0] for row in p_rows]
    p_mix_1 = [row[2] for row in p_rows]  # 多模态
    p_img_1 = [row[3] for row in p_rows]  # 图像
    p_audio_1 = [row[4] for row in p_rows]  # 音频
    p_baseline_1 = [row[5] for row in p_rows]  # 音频
    seq_map = {
        0: 0,
        1: 0.1,
        2: 0.2,
        3: 0.25,
        4: 0.3,
        5: 0.35,
        6: 0.4,
        7: 0.45,
        8: 0.5,
        9: 0.55,
        10: 0.60,
        11: 0.65,
        12: 0.7,
        13: 0.75,
        14: 0.8,
        15: 0.9,
        16: 1,
    }
    ratio_x = [seq_map[sid] for sid in seq_ids]

    plt.figure(figsize=(10, 6))

    # 获取拟合参数
    _, k_mix = plot_sigmoid(ratio_x, p_mix_1, "mix", "tab:blue")
    _, k_img = plot_sigmoid(ratio_x, p_img_1, "image", "tab:orange")
    _, k_audio = plot_sigmoid(ratio_x, p_audio_1, "audio", "tab:green")
    _, k_baseline = plot_sigmoid(
        ratio_x, p_baseline_1, "baseline (Bayes Logit)", "tab:red"
    )

    # === 验证贝叶斯最优整合 ===
    if k_img is not None and k_audio is not None and k_mix is not None:
        # 计算理论上的贝叶斯最优斜率
        k_optimal = np.sqrt(k_img**2 + k_audio**2)

        print("\n========== 贝叶斯整合验证 ==========")
        print(f"Image Slope (k_v): {k_img:.4f}")
        print(f"Audio Slope (k_a): {k_audio:.4f}")
        print(f"Actual Mix Slope : {k_mix:.4f}")
        print(f"Baseline Slope : {k_baseline:.4f}")
        print(f"Optimal Bayes Slope (sqrt(kv^2 + ka^2)): {k_optimal:.4f}")

        diff = abs(k_mix - k_optimal) / k_optimal * 100
        print(f"偏差: {diff:.2f}%")
        bayes_path = os.path.join(cfg.ckpt_dir, "bayes.json")
        with open(bayes_path, "w", encoding="utf-8") as f:
            # 将对象转为字典保存，过滤掉方法
            config_dict = {
                "Image Slope (k_v)": f"{k_img:.4f}",
                "Audio Slope (k_a)": f"{k_audio:.4f}",
                "Actual Mix Slope": f"{k_mix:.4f}",
                "Baseline Slope": f"{k_baseline:.4f}",
                "Optimal Bayes Slope (sqrt(kv^2 + ka^2))": f"{k_optimal:.4f}",
                "偏差": f"{diff:.2f}%",
            }
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

    plt.xlabel("dog ratio")
    plt.ylabel("Proportion dog choice")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(np.arange(0, 1.1, 0.1))  # 从 0 到 1，每 0.1 为一个刻度
    plt.legend()

    plt.tight_layout()
    img_save = os.path.join(cfg.ckpt_dir, "bayes_result.png")
    plt.savefig(img_save, dpi=300)
    
    from mix_dataset import PairedVisionAudioDataset
    
    dataset_idx = [0, 3, 8, 13, 16]  # 选择不同混合比例的 seq_id 进行绘图
    img_map = {0: "0_00", 3: "0_25", 8: "0_50", 13: "0_75", 16: "1_00"}
    audio_map = {0: "80", 3: "10", 8: "0", 13: "-10", 16: "-80"}
    
    for idx in dataset_idx:
        img_root = os.path.join(img_data_dir, f"dogs_{idx}_ratio_{img_map[idx]}")
        audio_root = os.path.join(audio_data_dir, f"cats_{idx}_snrdb_{audio_map[idx]}")
        # 假设 'dog' 类别标签为 1, 'cat' 类别为 0
        paired_dataset = PairedVisionAudioDataset(
            image_data_dir=img_root,
            audio_data_dir=audio_root,
            image_transforms=_val_transforms,
            audio_feature_extractor=feature_extractor
        )

        # 2. 绘制 "狗" 的 Logit 分布图
        logit_plot, dog_stats = plot_logit_distribution(
            inference_model,
            img_model_best,
            audio_model_best,
            paired_dataset,
            device,
            class_name="dog",
            label_id=1,
        )
        logit_plot_save_path = os.path.join(cfg.ckpt_dir, "logits_dist", f"logit_dist_dog_{idx}.png")
        os.makedirs(os.path.dirname(logit_plot_save_path), exist_ok=True)
        logit_plot.savefig(logit_plot_save_path, dpi=300)
        print(f"Logit 分布图已保存至: {logit_plot_save_path}")

        # 将统计结果保存到 JSON 文件
        all_stats = {"dog_class_stats": dog_stats}
        stats_save_path = os.path.join(cfg.ckpt_dir, "logits_dist", f"logit_stats_{idx}.json")
        with open(stats_save_path, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, indent=4)
        print(f"Logit 统计数据已保存至: {stats_save_path}")
