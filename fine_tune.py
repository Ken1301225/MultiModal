import os
import librosa
import torch
import torch

os.environ["HF_DATASETS_OFFLINE"] = "1"

from transformers import AutoModelForImageClassification

# os.environ["WANDB_API_KEY"] = "ea6035b7de02f0fc022ba881d1520987a321ede1"
# from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
from datasets import config

config.TORCHCODEC_AVAILABLE = False
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datasets import Dataset, Image, ClassLabel, Features
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    AutoModel,
)
import torch
from datasets import load_dataset, Audio
import evaluate


cache_dir = "/home/amax/dakai/dataset/microsoft"
img_dataset = load_dataset("microsoft/cats_vs_dogs", cache_dir=cache_dir)
img_dataset
img_dataset_split = img_dataset["train"].train_test_split(
    test_size=0.3, train_size=0.7, seed=42
)

img_train_ds = img_dataset_split["train"]
img_val_ds = img_dataset_split["test"]

from transformers import AutoImageProcessor
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
)

img_model_checkpoint = "microsoft/resnet-50"
cache_dir = "/home/amax/dakai/neuron/model/restnet50"
img_cache_checkpoint = "/home/amax/.cache/huggingface/hub/models--microsoft--resnet-50/snapshots/34c2154c194f829b11125337b98c8f5f9965ff19"
image_processor = AutoImageProcessor.from_pretrained(img_cache_checkpoint)

# 2. 定义转换函数
# 图像均值和方差来自 ImageNet
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

# 训练集转换 (包含数据增强)
_train_transforms = Compose(
    [
        Resize((256, 256)),
        RandomHorizontalFlip(),
        CenterCrop(224),
        ToTensor(),
        normalize,
    ]
)

# 验证集转换 (仅调整大小和归一化)
_val_transforms = Compose(
    [
        Resize((256, 256)),
        CenterCrop(224),
        ToTensor(),
        normalize,
    ]
)


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


img_model = AutoModelForImageClassification.from_pretrained(
    img_cache_checkpoint,  # "microsoft/resnet-50"
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # 忽略输出层大小不匹配 (从 ImageNet 1000类 -> 2类)
)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# ============ 6. 数据整理器 (Data Collator) ============
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example[label_col] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# ============ 7. 配置训练参数 ============
training_args = TrainingArguments(
    output_dir="./resnet-cats-dogs3",
    remove_unused_columns=False,  # ⚠️ 关键！必须设为 False，否则 'image' 列会被删除导致 transform 失败
    eval_strategy="epoch",  # 每个 epoch 评估一次
    save_strategy="epoch",  # 每个 epoch 保存一次
    learning_rate=5e-4,
    per_device_train_batch_size=16,  # 如果显存不够，改小这个数字 (如 16)
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="/home/amax/dakai/neuron/logs/finetune2",  # TensorBoard 日志存放路径
    report_to=["tensorboard"],
)

# ============ 8. 开始训练 ============
trainer = Trainer(
    model=img_model,
    args=training_args,
    train_dataset=img_train_ds,
    eval_dataset=img_val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

print("开始训练...")
trainer.train()


# # 1. 加载本地音频数据集 (不使用 Audio 特性)
# audio_data_dir = "/home/amax/dakai/dataset/dc_w/DvC"


# def load_local_audio_dataset(data_dir):
#     audio_files = []
#     labels = []

#     for label_name in os.listdir(data_dir):
#         label_dir = os.path.join(data_dir, label_name)

#         if not os.path.isdir(label_dir) or label_name == "desktop.ini":
#             continue

#         print(f"加载 {label_name} 类别...")

#         for audio_file in os.listdir(label_dir):
#             if audio_file.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
#                 audio_path = os.path.join(label_dir, audio_file)
#                 audio_files.append(audio_path)
#                 labels.append(label_name.lower())

#     print(f"共加载 {len(audio_files)} 个音频文件")
#     print(f"类别: {set(labels)}")

#     # ⚠️ 不要使用 Audio(),直接保存路径
#     dataset = Dataset.from_dict(
#         {"audio_path": audio_files, "label": labels}  # 改名为 audio_path
#     )

#     # 将标签转换为整数
#     unique_labels = sorted(set(labels))
#     label2id = {label: i for i, label in enumerate(unique_labels)}

#     def encode_labels(example):
#         example["label"] = label2id[example["label"]]
#         return example

#     dataset = dataset.map(encode_labels)

#     return dataset, label2id


# audio_dataset, label2id_audio = load_local_audio_dataset(audio_data_dir)
# id2label_audio = {i: label for label, i in label2id_audio.items()}

# print(f"\n标签映射: {label2id_audio}")
# print(audio_dataset)


# audio_data_split = audio_dataset.train_test_split(0.2, 0.8, seed=42)
# audio_train_ds = audio_data_split["train"]
# audio_val_ds = audio_data_split["test"]


# audio_model_name = "facebook/wav2vec2-base"
# cache_dir = "/home/amax/dakai/neuron/model/facebook"
# cache_checkpoint = "/home/amax/dakai/neuron/model/facebook/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
# feature_extractor = AutoFeatureExtractor.from_pretrained(cache_checkpoint)


# def preprocess_function(examples):

#     audio_arrays = []

#     # 从文件路径加载音频
#     for audio_path in examples["audio_path"]:
#         audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)
#         audio_arrays.append(audio_array)

#     inputs = feature_extractor(
#         audio_arrays,
#         sampling_rate=16000,
#         padding=True,
#         max_length=16000 * 10,
#         truncation=True,
#     )
#     inputs["labels"] = examples["label"]
#     return inputs


# audio_train_ds = audio_train_ds.map(
#     preprocess_function, remove_columns=["audio_path"], batched=True  # 改为 audio_path
# )
# audio_val_ds = audio_val_ds.map(
#     preprocess_function, remove_columns=["audio_path"], batched=True
# )


# audio_model = AutoModelForAudioClassification.from_pretrained(
#     cache_checkpoint,
#     num_labels=len(label2id_audio),
#     label2id=label2id_audio,  # 之前写错了
#     id2label=id2label_audio,
#     ignore_mismatched_sizes=True,
#     cache_dir=cache_dir,
# )

# print("模型加载完成!")
# print(f"训练集大小: {len(audio_train_ds)}")
# print(f"验证集大小: {len(audio_val_ds)}")

# # 评估指标
# accuracy = evaluate.load("accuracy")


# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels)


# # 数据整理器
# def audio_collate_fn(examples):
#     import torch.nn.utils.rnn as rnn_utils

#     input_values = [torch.tensor(ex["input_values"]).squeeze() for ex in examples]
#     input_values = rnn_utils.pad_sequence(input_values, batch_first=True)
#     labels = torch.tensor([ex["labels"] for ex in examples])
#     return {"input_values": input_values, "labels": labels}


# # 训练参数
# audio_training_args = TrainingArguments(
#     output_dir="./wav2vec2-cats-dogs",
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=3e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=5,
#     warmup_ratio=0.1,
#     logging_steps=10,
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
#     report_to="none",
# )

# # 训练
# audio_trainer = Trainer(
#     model=audio_model,
#     args=audio_training_args,
#     train_dataset=audio_train_ds,
#     eval_dataset=audio_val_ds,
#     tokenizer=feature_extractor,
#     compute_metrics=compute_metrics,
#     data_collator=audio_collate_fn,
# )

# print("开始训练音频分类模型...")
# # audio_trainer.train()
