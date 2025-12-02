import os
import random
import torch
import librosa
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset, load_from_disk
from PIL import Image as PILImage
from transformers import AutoImageProcessor
from transformers import AutoFeatureExtractor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def load_local_image_dataset(data_dir):
    """
    修改版：加载之前通过 save_to_disk 保存的混合图像数据集 (Arrow格式)。
    此函数替代了原先解析 [label].[number].jpg 文件名的逻辑。
    """
    print(f"正在从 {data_dir} 加载混合图像数据集...")

    # 1. 核心修改：直接使用 load_from_disk 读取数据
    # 混合数据集已经是标准的 Dataset 对象，不需要再解析文件名
    try:
        dataset = load_from_disk(data_dir)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"找不到数据集路径: {data_dir}。请确保指向的是 save_to_disk 生成的文件夹。"
        )
    except Exception as e:
        raise RuntimeError(f"加载数据集失败: {e}")

    # 2. 列名适配：适配 create_paired_test_set 函数
    # 原有的混合代码生成的列名是 'label' (单数)，但配对函数读取的是 'labels' (复数)
    if "label" in dataset.column_names and "labels" not in dataset.column_names:
        dataset = dataset.rename_column("label", "labels")
        print("已自动将列名 'label' 重命名为 'labels' 以适配配对函数。")

    # 3. 格式检查
    print(f"共加载 {len(dataset)} 个图像样本")

    # 打印一下标签示例，确认是整数 (0, 1) 而不是字符串
    # 音频加载函数的 label 是 int 类型，这里必须也是 int 才能配对成功
    if len(dataset) > 0:
        sample_label = dataset[0]["labels"]
        print(f"样本标签示例: {sample_label} (类型: {type(sample_label)})")
        # 混合数据集通常包含 0(猫) 和 1(狗)，直接对应音频的 id

    return dataset


def load_local_audio_dataset(data_dir):
    audio_files = []
    labels = []

    # for label_name in os.listdir(data_dir):
    #     label_dir = os.path.join(data_dir, label_name)

    #     if not os.path.isdir(label_dir) or label_name == "desktop.ini":
    #         continue

    #     print(f"加载 {label_name} 类别...")

    #     for audio_file in os.listdir(label_dir):
    #         if audio_file.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
    #             audio_path = os.path.join(label_dir, audio_file)
    #             audio_files.append(audio_path)
    #             labels.append(label_name.lower())
    data_dir_path = Path(data_dir)
    dir_name = data_dir_path.name  # e.g., 'cats_0_snrdb_80'
    parts = dir_name.split("_")

    if parts[0] == "cats":
        if float(parts[3]) > 0:
            label_name = "cats"
        elif float(parts[3]) == 0:
            choice = random.randint(0, 1)
            if choice == 0:
                label_name = "cats"
            else:
                label_name = "dogs"
        else:
            label_name = "dogs"

    for audio_path in data_dir_path.iterdir():
        if audio_path.suffix in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            # 将 Path 对象转换为字符串存储
            audio_files.append(str(audio_path))
            labels.append(label_name.lower())

    print(f"共加载 {len(audio_files)} 个音频文件")
    print(f"类别: {set(labels)}")

    dataset = Dataset.from_dict(
        {"audio_path": audio_files, "label": labels}  # 改名为 audio_path
    )

    label2id = {"cats": 0, "dogs": 1}

    def encode_labels(example):
        example["label"] = label2id[example["label"]]
        return example

    dataset = dataset.map(encode_labels)

    return dataset, label2id


class PairedVisionAudioDataset(TorchDataset):
    """
    一个继承自 TorchDataset 的数据集类，用于随机配对图像和音频数据。

    该类会加载本地的图像和音频数据集，并根据标签进行随机配对。
    数据集的长度由图像数据集的样本数决定（优先满足图像配对）。
    """

    def __init__(
        self,
        image_data_dir,
        audio_data_dir,
        image_transforms,
        audio_feature_extractor,
        audio_sampling_rate=16000,
    ):
        """
        初始化数据集。
        Args:
            image_data_dir (str): 图像数据集的文件夹路径 (由 save_to_disk 生成)。
            audio_data_dir (str): 音频数据集的文件夹路径。
            image_transforms (callable): 应用于图像的转换函数。
            audio_feature_extractor (callable): 用于处理音频的特征提取器。
            audio_sampling_rate (int): 音频文件的目标采样率。
        """
        super().__init__()
        self.image_transforms = image_transforms
        self.audio_feature_extractor = audio_feature_extractor
        self.audio_sampling_rate = audio_sampling_rate

        # 1. 使用你提供的函数加载数据集
        self.image_ds = load_local_image_dataset(image_data_dir)
        self.audio_ds, _ = load_local_audio_dataset(audio_data_dir)

        # 2. 按标签对音频数据进行分组，以便快速随机抽样
        print("正在按标签对音频数据进行分组以便配对...")
        self.audio_by_label = {0: [], 1: []}  # 假设标签为 0:cats, 1:dogs
        for item in self.audio_ds:
            label = item["label"]
            if label in self.audio_by_label:
                self.audio_by_label[label].append(item)

        if not self.audio_by_label[0] and not self.audio_by_label[1]:
            raise ValueError("音频数据集中未找到任何有效标签的数据，无法进行配对。")
        print("音频数据分组完成。")

    def __len__(self):
        """返回数据集的长度，以图像数据集为准。"""
        return len(self.image_ds)

    def __getitem__(self, idx):
        """
        获取一个配对的数据样本。
        """
        # 1. 获取图像样本及其标签
        image_item = self.image_ds[idx]
        image: PILImage.Image = image_item["image"]
        label = image_item["labels"]  # 图像数据集的标签列名为 'labels'

        # 2. 根据图像标签，随机选择一个匹配的音频样本
        # 检查该标签是否有对应的音频文件
        if self.audio_by_label.get(label):
            audio_pool = self.audio_by_label[label]
        else:
            # 如果当前标签没有音频（例如，测试一个只有猫图像的数据集，但音频文件夹只有狗的声音），
            # 则从另一个标签的音频池中随机选择，以确保程序不会崩溃。
            other_label = 1 - label
            if not self.audio_by_label.get(other_label):
                raise RuntimeError(
                    f"无法为标签 {label} 找到任何音频样本，并且备用标签 {other_label} 也没有音频。"
                )
            audio_pool = self.audio_by_label[other_label]

        selected_audio_item = random.choice(audio_pool)
        audio_path = selected_audio_item["audio_path"]

        # 3. 对图像和音频进行预处理
        # 处理图像
        pixel_values = self.image_transforms(image)

        # 处理音频
        audio_array, _ = librosa.load(
            audio_path, sr=self.audio_sampling_rate, mono=True
        )
        # 特征提取器返回一个字典，我们需要解包并取 'input_values'
        audio_inputs = self.audio_feature_extractor(
            audio_array,
            sampling_rate=self.audio_sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        # 通常 input_values 是一个 [1, N] 的张量，我们用 squeeze() 去掉批次维度
        input_values = audio_inputs["input_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ==============================================================================
#  方案一：数据集工厂 (迭代器模式)
# ==============================================================================


class PairedDatasetFactory:
    """
    一个数据集工厂，用于扫描并匹配图像和音频的主目录下的子数据集。

    它本身不是一个数据集，而是一个迭代器。当你遍历它时，它会为每一对
    匹配的子文件夹（按序号匹配）生成一个 PairedVisionAudioDataset 实例。
    """

    def __init__(
        self, main_image_dir, main_audio_dir, image_transforms, audio_feature_extractor
    ):
        """
        初始化工厂。
        Args:
            main_image_dir (str): 包含多个图像子文件夹的主目录。
            main_audio_dir (str): 包含多个音频子文件夹的主目录。
            image_transforms (callable): 应用于图像的转换函数。
            audio_feature_extractor (callable): 用于处理音频的特征提取器。
        """
        self.image_transforms = image_transforms
        self.audio_feature_extractor = audio_feature_extractor

        print("--- PairedDatasetFactory 初始化 ---")
        # 1. 扫描并按序号索引子文件夹
        image_subfolders = self._scan_subfolders(main_image_dir)
        audio_subfolders = self._scan_subfolders(main_audio_dir)

        # 2. 找到共有的序号并存储匹配的路径对
        self.matched_paths = []
        shared_sequence_ids = sorted(
            list(set(image_subfolders.keys()) & set(audio_subfolders.keys()))
        )

        if not shared_sequence_ids:
            print("警告: 在图像和音频主目录中没有找到任何序号匹配的子文件夹。")

        for seq_id in shared_sequence_ids:
            self.matched_paths.append(
                {
                    "sequence_id": seq_id,
                    "image_dir": image_subfolders[seq_id],
                    "audio_dir": audio_subfolders[seq_id],
                }
            )

        print(
            f"成功匹配 {len(self.matched_paths)} 对子数据集，序号为: {shared_sequence_ids}"
        )
        print(self.matched_paths[0])
        print("------------------------------------")

    def _scan_subfolders(self, main_dir):
        """辅助函数，扫描目录并根据序号构建字典。"""
        subfolders_by_seq = {}
        main_dir_path = Path(main_dir)  # 转换为 Path 对象

        if not main_dir_path.is_dir():
            print(f"警告: 主目录不存在: {main_dir}")
            return {}

        for path in main_dir_path.iterdir():  # 遍历目录中的所有项
            if path.is_dir():  # 检查是否为文件夹
                try:
                    # e.g., 'dogs_2_ratio_0_20' -> 2
                    seq_id = int(path.name.split("_")[1])
                    subfolders_by_seq[seq_id] = str(path)
                except (IndexError, ValueError):
                    print(f"警告: 无法从 '{path.name}' 中解析序号，已跳过。")
        return subfolders_by_seq

    def __len__(self):
        """返回匹配的数据集对的数量。"""
        return len(self.matched_paths)

    def __iter__(self):
        """使类成为一个迭代器。"""
        self.n = 0
        return self

    def __next__(self):
        """返回下一个 PairedVisionAudioDataset 实例。"""
        if self.n < len(self.matched_paths):
            pair_info = self.matched_paths[self.n]
            self.n += 1

            print(f"\n=> 正在创建序号为 {pair_info['sequence_id']} 的数据集...")

            dataset = PairedVisionAudioDataset(
                image_data_dir=pair_info["image_dir"],
                audio_data_dir=pair_info["audio_dir"],
                image_transforms=self.image_transforms,
                audio_feature_extractor=self.audio_feature_extractor,
            )
            # 返回数据集实例和它的序号
            return dataset, pair_info["sequence_id"]
        else:
            raise StopIteration


# ==============================================================================
#  方案二：聚合数据集 (单一数据集模式)
# ==============================================================================


class MegaPairedDataset(TorchDataset):
    """
    一个聚合的数据集，将多个 PairedVisionAudioDataset 实例整合成一个。

    它表现得像一个单一的、巨大的数据集。在 __getitem__ 时，除了返回数据样本，
    还会额外返回一个 'sequence_id'，以标识样本的来源子数据集。
    """

    def __init__(
        self, main_image_dir, main_audio_dir, image_transforms, audio_feature_extractor
    ):
        super().__init__()
        print("--- MegaPairedDataset 初始化 ---")
        # 1. 使用 Factory 找到所有匹配的子数据集并实例化它们
        factory = PairedDatasetFactory(
            main_image_dir, main_audio_dir, image_transforms, audio_feature_extractor
        )

        self.sub_datasets = []
        self.seq_ids = []
        # 遍历工厂，收集所有实例化的子数据集
        for dataset, seq_id in factory:
            self.sub_datasets.append(dataset)
            self.seq_ids.append(seq_id)

        # 2. 计算累积长度，用于全局索引
        self.cumulative_sizes = self.cumsum(self.sub_datasets)
        print("---------------------------------")

    @staticmethod
    def cumsum(sequence):
        """计算数据集长度的累积和。"""
        r, s = [], 0
        for d in sequence:
            l = len(d)
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        """返回所有子数据集合并后的总长度。"""
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        """
        根据全局索引获取一个样本。
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        # 查找该索引属于哪个子数据集
        dataset_idx = torch.searchsorted(
            torch.tensor(self.cumulative_sizes), idx, right=True
        ).item()

        # 计算在该子数据集内的局部索引
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # 从子数据集中获取样本
        sample = self.sub_datasets[dataset_idx][sample_idx]

        # 添加来源信息
        sample["sequence_id"] = self.seq_ids[dataset_idx]

        return sample


# --- 如何使用这个类的示例 ---
if __name__ == "__main__":

    # # 1. 定义图像和音频的路径及处理器
    # IMG_DATA_DIR = "/home/tomoon/datasets/dog_cat_img_cf/mixed_data1/mixed_cats_dogs_img/dogs_14_ratio_0_80" # 指向 save_to_disk 生成的文件夹
    # AUDIO_DATA_DIR = "/home/tomoon/datasets/dog_cat_audio_cf/dog_cat_wav/mix_test_set/cats_14_snrdb_-20"

    # # 图像预处理器
    # image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    # image_transforms = Compose([
    #     Resize((224, 224)),
    #     ToTensor(),
    #     Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    # ])
    # # 音频特征提取器
    # audio_feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    # # 2. 实例化数据集
    # paired_dataset = PairedVisionAudioDataset(
    #     image_data_dir=IMG_DATA_DIR,
    #     audio_data_dir=AUDIO_DATA_DIR,
    #     image_transforms=image_transforms,
    #     audio_feature_extractor=audio_feature_extractor
    # )

    # print(f"配对数据集大小: {len(paired_dataset)}")
    # print("示例数据项:")
    # for i in range(5):
    #     sample = paired_dataset[i]
    #     print("图像像素值形状:", sample['pixel_values'].shape)
    #     print("音频输入值形状:", sample['input_values'].shape)
    #     print("标签:", sample['labels'])

    # # 3. 创建 DataLoader
    # from torch.utils.data import DataLoader
    # data_loader = DataLoader(paired_dataset, batch_size=8, shuffle=True)

    # # 4. 迭代检查数据
    # batch = next(iter(data_loader))
    # print("Batch keys:", batch.keys())
    # print("Pixel values shape:", batch['pixel_values'].shape)
    # print("Input values shape:", batch['input_values'].shape)
    # print("Labels:", batch['labels'])

    MAIN_IMG_DIR = "E:/NeuralScience/project/mixed_cats_dogs_img"
    MAIN_AUDIO_DIR = "E:/NeuralScience/project/DvC/mix_test_set"

    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    image_transforms = Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )
    audio_feature_extractor = AutoFeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base"
    )

    print("\n\n" + "=" * 20 + " 测试方案一: PairedDatasetFactory " + "=" * 20)
    dataset_factory = PairedDatasetFactory(
        main_image_dir=MAIN_IMG_DIR,
        main_audio_dir=MAIN_AUDIO_DIR,
        image_transforms=image_transforms,
        audio_feature_extractor=audio_feature_extractor,
    )

    # 遍历工厂，对每个生成的数据集进行操作
    for paired_dataset, seq_id in dataset_factory:
        print(f"--> 成功获取序号为 {seq_id} 的数据集，大小为: {len(paired_dataset)}")
        # 在这里你可以为每个数据集创建一个 DataLoader 并进行测试
        # test_loader = DataLoader(paired_dataset, batch_size=8)
        # ... run_evaluation(model, test_loader) ...
        if len(paired_dataset) > 0:
            sample = paired_dataset[0]
            print(f"    样本 'pixel_values' 形状: {sample['pixel_values'].shape}")
            print(f"    样本 'input_values' 形状: {sample['input_values'].shape}")
            print(f"    样本 'labels': {sample['labels']}")
        label_0 = paired_dataset[0]["labels"]
        for item in paired_dataset:
            label_1 = item["labels"]
            if label_0 != label_1:
                print(f"在第{seq_id}组发现不同标签的样本！")

    print("\n\n" + "=" * 20 + " 测试方案二: MegaPairedDataset " + "=" * 20)
    mega_dataset = MegaPairedDataset(
        main_image_dir=MAIN_IMG_DIR,
        main_audio_dir=MAIN_AUDIO_DIR,
        image_transforms=image_transforms,
        audio_feature_extractor=audio_feature_extractor,
    )

    print(f"\n聚合数据集总大小: {len(mega_dataset)}")
    if len(mega_dataset) > 0:
        print("从聚合数据集中随机抽取几个样本进行检查:")
        indices_to_check = [0, len(mega_dataset) // 2, len(mega_dataset) - 1]
        for i in indices_to_check:
            sample = mega_dataset[i]
            print(
                f"  - 样本索引 {i}: "
                f"来自序号 {sample['sequence_id']} 的子数据集, "
                f"图像像素值形状 {sample['pixel_values'].shape}, "
                f"音频输入值形状 {sample['input_values'].shape}, "
                f"标签为 {sample['labels'].item()}"
            )

    # 你可以为这个聚合数据集创建一个 DataLoader
    # from torch.utils.data import DataLoader
    # mega_loader = DataLoader(mega_dataset, batch_size=16, shuffle=True)
    # batch = next(iter(mega_loader))
    # print("\n一个批次的数据键:", batch.keys())
    # print("批次中的 sequence_ids:", batch['sequence_id'])
