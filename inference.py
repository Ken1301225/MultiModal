import os
import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoImageProcessor


os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

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
        all_logits_c = all_logits_v + all_logits_a  # Logit 加和融合

        def calc_prop(logits):
            # argmax -> 0 or 1 -> mean
            return (torch.argmax(logits, dim=1) == 1).float().mean().item()

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
        p_baseline_1 = calc_prop(all_logits_c)

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

    set_seed(42)

    img_model_cache = "/home/tomoon/.cache/huggingface/hub/models--microsoft--resnet-50/snapshots/34c2154c194f829b11125337b98c8f5f9965ff19"
    image_processor = AutoImageProcessor.from_pretrained(img_model_cache, use_fast=True)

    audio_model_cache = "/home/tomoon/codes/lecture_project/Neural Science/models/wav2vec2-base/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        audio_model_cache, use_fast=True
    )

    audio_data_dir = "/home/tomoon/datasets/dog_cat_audio_cf/dog_cat_wav/mix_test_set"
    img_data_dir = "/home/tomoon/datasets/dog_cat_img_cf/mixed_data1/mixed_cats_dogs_img"

    _train_transforms, _val_transforms = get_transforms()

    multi__ds = PairedDatasetFactory(
        img_data_dir, audio_data_dir, _train_transforms, feature_extractor
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_model_path = "/home/tomoon/codes/lecture_project/Neural Science/models/resnet-cats-dogs3/checkpoint-4100"
    img_model_best = AutoModelForImageClassification.from_pretrained(img_model_path)
    audio_best_path = "/home/tomoon/codes/lecture_project/Neural Science/models/wav2vec2-cats-dogs/checkpoint-525"
    audio_model_best = AutoModelForAudioClassification.from_pretrained(audio_best_path)
    save_path = "./checkpoints/4100_525/attn/multimodal_attn_model_best6.pth"
    # save_path = "/home/amax/dakai/neuron/checkpoints/4100_525/drop/multimodal_attn_drop_model_best8.pth"

    img_model_best.to(device)
    audio_model_best.to(device)
    # inference_model = HumanLikeMultimodalModel(img_model_best, audio_model_best).to(
    #     device
    # )
    # inference_model = MultimodalModelDrop(img_model_best, audio_model_best).to(device)
    # inference_model = MultiModalAttnModel(img_model_best, audio_model_best, shared_dim=1024, attn_heads=16).to(device)
    inference_model = MultiModalAttnCLSModel(img_model_best, audio_model_best, shared_dim=1024, attn_heads=16).to(device)

    inference_model.load_state_dict(torch.load(save_path, map_location=device))
    inference_model.eval()
    print("模型加载成功！")

    p_rows = inference(
        inference_model,
        img_model_best,
        audio_model_best,
        multi__ds,
        device,
        mode=["mix", "image", "audio"],
        eval_mode=["choice_proportion"],
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

    plt.xlabel("dog ratio")
    plt.ylabel("Proportion dog choice")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(np.arange(0, 1.1, 0.1))  # 从 0 到 1，每 0.1 为一个刻度
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        "./img/attn/multimodal_attn_model_best6.png", dpi=300
    )
