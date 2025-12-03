import os
import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataset import (
    get_transforms,
    preprocess_function,
    load_local_audio_dataset_dir,
    RandomPairedDataset,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForAudioClassification,
    AutoModelForImageClassification,
)
import os
from datasets import load_dataset
from model import (
    HumanLikeMultimodalModel,
    print_trainable_parameters,
    MultimodalModelDrop,
    MultiModalAttnModel,
    MultiModalAttnCLSModel
)
from torch.utils.data import DataLoader
import torch.optim as optim


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


log_dir = "./logs/attn_1"  # 日志存储路径
writer = SummaryWriter(log_dir=log_dir)

set_seed(42)

img_model_path = "./resnet-cats-dogs3/checkpoint-4100"
img_model_best = AutoModelForImageClassification.from_pretrained(img_model_path)


device = "cuda" if torch.cuda.is_available() else "cpu"

audio_best_path = "./wav2vec2-cats-dogs/checkpoint-525"
audio_model_best = AutoModelForAudioClassification.from_pretrained(audio_best_path)


audio_data_dir = "E:/NeuralScience/project/DvC"

audio_dataset, label2id_audio = load_local_audio_dataset_dir(audio_data_dir)
id2label_audio = {i: label for label, i in label2id_audio.items()}

print(f"\n标签映射: {label2id_audio}")
print(audio_dataset)


audio_data_split = audio_dataset.train_test_split(0.2, 0.8, seed=42)
audio_train_ds = audio_data_split["train"]
audio_val_ds = audio_data_split["test"]


audio_train_ds = audio_train_ds.map(
    preprocess_function, remove_columns=["audio_path"], batched=True  # 改为 audio_path
)
audio_val_ds = audio_val_ds.map(
    preprocess_function, remove_columns=["audio_path"], batched=True
)


cache_dir = "E:/NeuralScience/project/img_dataset"
img_dataset = load_dataset("microsoft/cats_vs_dogs", cache_dir=cache_dir)
print(img_dataset)
img_dataset_split = img_dataset["train"].train_test_split(0.3, 0.7, seed=42)

img_train_ds = img_dataset_split["train"]
img_val_ds = img_dataset_split["test"]

_train_transforms, _val_transforms = get_transforms()


def train_transforms_fn(examples):
    examples["pixel_values"] = [
        _train_transforms(image.convert("RGB")) for image in examples["image"]
    ]
    return examples


def val_transforms_fn(examples):
    examples["pixel_values"] = [
        _val_transforms(image.convert("RGB")) for image in examples["image"]
    ]
    return examples


# 动态处理
img_train_ds.set_transform(train_transforms_fn)
img_val_ds.set_transform(val_transforms_fn)

label_col = "labels" if "labels" in img_dataset["train"].features else "label"
labels = img_dataset["train"].features[label_col].names
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

print(f"标签列名: {label_col}")
print(f"标签映射: {label2id}")


# multimodal_model = HumanLikeMultimodalModel(img_model_best, audio_model_best).to(device)
# multimodal_model = MultimodalModelDrop(
#     img_model_best,
#     audio_model_best,
#     shared_dim=512,
#     num_classes=2,
#     vision_drop_prob=0.5,  # 20% 概率整路屏蔽视觉
#     audio_drop_prob=0.5,  # 20% 概率整路屏蔽音频
#     emb_mask_prob=0.7,  # 每个 embedding 维度 10% 概率置零
# ).to(device)
multimodal_model = MultiModalAttnCLSModel(
    img_model_best,
    audio_model_best,
    shared_dim=512,
    num_classes=2,
    attn_heads=8,
    attn_dropout=0.4,
    vision_drop_prob=0.3,
    audio_drop_prob=0.3,
).to(device)
# multimodal_model = MultiModalAttnCLSModel(
#     img_model_best,
#     audio_model_best,
#     shared_dim=512,
#     num_classes=2,
#     attn_heads=8,
#     attn_dropout=0.4,
#     vision_drop_prob=0.3,
#     audio_drop_prob=0.3,
# )
multi_train_ds = RandomPairedDataset(img_train_ds, audio_train_ds)
multi_val_ds = RandomPairedDataset(img_val_ds, audio_val_ds)


train_loader = DataLoader(multi_train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(multi_val_ds, batch_size=16, shuffle=False)

optimizer = optim.AdamW(multimodal_model.parameters(), lr=1e-5)


print("开始多模态协同训练...")

best_val_acc = 0.0
best_val_loss = np.inf
save_path = (
    "./checkpoints/4100_525/attn/multimodal_attn_model_best1.pth"
)

# print_trainable_parameters(multimodal_model)

for epoch in range(10):
    # ================= 训练阶段 =================
    multimodal_model.train()
    total_train_loss = 0

    # 使用 tqdm 显示训练进度
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for step, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(device)
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = multimodal_model(pixel_values, input_values, labels)
        loss = outputs["loss"]

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        # 更新进度条上的 loss 显示
        progress_bar.set_postfix({"loss": loss.item()})

        global_step = epoch * len(train_loader) + step

        writer.add_scalar("Train/Loss", loss.item(), global_step)

    avg_train_loss = total_train_loss / len(train_loader)

    # ================= 验证阶段 =================
    multimodal_model.eval()
    total_val_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = multimodal_model(pixel_values, input_values, labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

            total_val_loss += loss.item()

            # 计算准确率
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = correct_predictions / total_samples
    writer.add_scalar("Val/Loss", avg_val_loss, epoch)
    writer.add_scalar("Val/Accuracy", val_acc, epoch)
    print(
        f"Epoch {epoch+1}: "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # ================= 保存最佳模型 =================
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(multimodal_model.state_dict(), save_path)
        print(f"  >>> 新的最佳模型已保存 (Acc: {best_val_loss:.4f})")

print(f"\n训练结束! 最佳验证集准确率: {best_val_acc:.4f}")
writer.close()
