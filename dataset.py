import os
import random
import librosa
import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
from transformers import AutoFeatureExtractor, AutoImageProcessor
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
)
import os
from datasets import load_from_disk, concatenate_datasets


class RandomPairedDataset(TorchDataset):
    def __init__(self, image_ds, audio_ds):
        self.image_ds = image_ds
        self.audio_ds = audio_ds

        self.image_indices_by_label = {}
        for idx, item in enumerate(image_ds):
            label = item.get("labels", item.get("label"))
            if label not in self.image_indices_by_label:
                self.image_indices_by_label[label] = []
            self.image_indices_by_label[label].append(idx)

        self.available_indices_by_label = {
            label: indices.copy()
            for label, indices in self.image_indices_by_label.items()
        }

    def __len__(self):
        return len(self.audio_ds)

    def __getitem__(self, idx):
        # 1. 获取图像数据
        # 因为 image_ds 已经 set_transform 了，这里直接取出来的就是包含 'pixel_values' 的字典
        audio_train_ds_item = self.audio_ds[idx]

        # 直接获取已经增强过的 Tensor
        audio_input = torch.tensor(audio_train_ds_item["input_values"])

        # 兼容不同的标签列名
        label = audio_train_ds_item.get("labels", audio_train_ds_item.get("label"))

        # 2. 随机匹配音频
        if (
            label in self.available_indices_by_label
            and len(self.available_indices_by_label[label]) > 0
        ):
            # 从可用索引中随机选择一个
            available_list = self.available_indices_by_label[label]
            image_idx = random.choice(available_list)
            # 从可用池中移除已选择的索引
            available_list.remove(image_idx)

            # 如果某个标签的可用图像用完了，重新填充
            if len(available_list) == 0:
                self.available_indices_by_label[label] = self.image_indices_by_label[
                    label
                ].copy()
                print(f"标签 {label} 的图像池已重置")
        else:
            # 如果该标签不存在，随机选择一个
            image_idx = random.randint(0, len(self.image_ds) - 1)
            print("使用随机索引（标签不匹配）")

        img_item = self.image_ds[image_idx]
        pixel_values = img_item["pixel_values"]

        return {
            "pixel_values": pixel_values,  # [3, 224, 224] (已增强)
            "input_values": audio_input,  # [seq_len]
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_local_audio_dataset_dir(data_dir):
    audio_files = []
    labels = []

    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)

        if not os.path.isdir(label_dir) or label_name == "desktop.ini":
            continue

        print(f"加载 {label_name} 类别...")

        for audio_file in os.listdir(label_dir):
            if audio_file.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                audio_path = os.path.join(label_dir, audio_file)
                audio_files.append(audio_path)
                labels.append(label_name.lower())

    print(f"共加载 {len(audio_files)} 个音频文件")
    print(f"类别: {set(labels)}")

    # ⚠️ 不要使用 Audio(),直接保存路径
    dataset = Dataset.from_dict(
        {"audio_path": audio_files, "label": labels}  # 改名为 audio_path
    )

    # 将标签转换为整数
    unique_labels = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}

    def encode_labels(example):
        example["label"] = label2id[example["label"]]
        return example

    dataset = dataset.map(encode_labels)

    return dataset, label2id


def preprocess_function(examples):
    audio_model_name = "facebook/wav2vec2-base"
    cache_dir = "./cache_dir/wav2vec2-base"
    # audio_model_cache = "/home/amax/dakai/neuron/model/facebook/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        audio_model_name, cache_dir=cache_dir, use_fast=True
    )

    audio_arrays = []

    # 从文件路径加载音频
    for audio_path in examples["audio_path"]:
        audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)
        audio_arrays.append(audio_array)

    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
        max_length=16000 * 10,
        truncation=True,
    )
    inputs["labels"] = examples["label"]
    return inputs


def get_transforms():
    # img_model_cache = "/home/amax/.cache/huggingface/hub/models--microsoft--resnet-50/snapshots/34c2154c194f829b11125337b98c8f5f9965ff19"
    img_model_checkpoint = "microsoft/resnet-50"
    cache_dir = "./cache_dir/restnet50"
    image_processor = AutoImageProcessor.from_pretrained(img_model_checkpoint, cache_dir=cache_dir, use_fast=True)

    normalize = Normalize(
        mean=image_processor.image_mean, std=image_processor.image_std
    )

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

    return _train_transforms, _val_transforms


def load_local_image_dataset(data_dir):
    """
    从指定的单个文件夹路径加载一个Hugging Face数据集。

    Args:
        data_dir (str): 数据集文件夹的完整路径。
                        例如: /home/amax/dakai/dataset/mix_imgset/mixed_cats_dogs_img/dogs_0_ratio_0_00

    Returns:
        datasets.Dataset: 加载的数据集，如果失败则返回 None。
    """
    print(f"正在从 {data_dir} 加载图像数据集...")

    # 检查路径是否存在且为目录
    if not os.path.isdir(data_dir):
        print(f"错误: 路径不是一个有效的目录 -> {data_dir}")
        return None

    try:
        # 直接从磁盘加载数据集
        dataset = load_from_disk(data_dir)
        print(
            f"  - 已成功加载: {os.path.basename(data_dir)} (包含 {len(dataset)} 个样本)"
        )
        return dataset
    except Exception as e:
        print(f"  - 加载 {data_dir} 失败: {e}")
        return None


def load_local_image_alldataset(base_dir):
    """
    从包含多个Hugging Face数据集子文件夹的基础目录中加载并合并所有图像数据集。

    Args:
        base_dir (str): 包含多个数据集子文件夹的路径。
                        例如: /home/amax/dakai/dataset/mix_imgset/mixed_cats_dogs_img

    Returns:
        datasets.Dataset: 一个合并了所有子数据集的大数据集。
    """
    all_datasets = []

    print(f"正在从基础目录 {base_dir} 加载并合并图像数据集...")

    # 1. 获取所有子文件夹的名称并排序
    try:
        subfolders = sorted([f.name for f in os.scandir(base_dir) if f.is_dir()])
    except FileNotFoundError:
        print(f"错误: 目录不存在 -> {base_dir}")
        return None

    if not subfolders:
        print(f"警告: 在 {base_dir} 中没有找到任何子文件夹。")
        return None

    # 2. 遍历每个子文件夹，使用 load_from_disk 加载
    for folder in subfolders:
        full_path = os.path.join(base_dir, folder)
        try:
            ds = load_from_disk(full_path)
            all_datasets.append(ds)
            print(f"  - 已成功加载: {folder} (包含 {len(ds)} 个样本)")
        except Exception as e:
            print(f"  - 加载 {folder} 失败: {e}")

    # 3. 如果成功加载了任何数据集，则将它们合并
    if all_datasets:
        combined_dataset = concatenate_datasets(all_datasets)
        print(f"\n所有图像子数据集已成功合并！总样本数: {len(combined_dataset)}")
        return combined_dataset
    else:
        print("\n未能加载任何有效的图像数据集。")
        return None


def load_local_audio_dataset(data_dir):
    audio_files = []
    labels = []

    parts_1 = data_dir.split("/")
    parts_2 = parts_1[-1].split("_")
    if parts_2[0] == "cats":
        if float(parts_2[3]) > 0:
            label_name = "cats"
        elif int(float(parts_2[3]) == 0):
            choice = random.randint(0, 1)
            if choice == 0:
                label_name = "cats"
            else:
                label_name = "dogs"
        else:
            label_name = "dogs"

    for audio_file in os.listdir(data_dir):
        if audio_file.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            audio_path = os.path.join(data_dir, audio_file)
            audio_files.append(audio_path)
            labels.append(label_name.lower())

    print(f"共加载 {len(audio_files)} 个音频文件")
    print(f"类别: {set(labels)}")

    # ⚠️ 不要使用 Audio(),直接保存路径
    dataset = Dataset.from_dict(
        {"audio_path": audio_files, "label": labels}  # 改名为 audio_path
    )

    label2id = {"cats": 0, "dogs": 1}

    def encode_labels(example):
        example["label"] = label2id[example["label"]]
        return example

    dataset = dataset.map(encode_labels)

    return dataset, label2id
