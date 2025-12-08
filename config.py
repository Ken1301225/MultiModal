import os
import datetime
import json


class Config:
    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 42)
        self.device = kwargs.get("device", "cuda")
        self.base_ckpt_dir = kwargs.get(
            "base_ckpt_dir", "/home/amax/dakai/neuron/checkpoints"
        )
        self.batch_size = kwargs.get("batch_size", 16)
        self.lr = kwargs.get("lr", 1e-5)
        self.epochs = kwargs.get("epochs", 10)
        self.optimizer_name = kwargs.get("optimizer_name", "AdamW")
        self.model_type = kwargs.get("model_type", "MultimodalModelDrop")
        self.model_params = kwargs.get(
            "model_params",
            {
                "shared_dim": 1024,
                "num_classes": 2,
                "attn_heads": 16,
                "attn_dropout": 0.5,
                "linear_dropout": 0.2,
                "vision_drop_prob": 0.5,
                "audio_drop_prob": 0.5,
                "emb_mask_prob": 0.7,
            },
        )

        self.run_name = self._generate_run_name()
        self.ckpt_dir = os.path.join(self.base_ckpt_dir, self.run_name)
        self.log_dir = os.path.join(self.ckpt_dir, "logs")
        self.save_path = os.path.join(self.ckpt_dir, "best_model.pth")
        self.img_model_path = kwargs.get(
            "img_model_path",
            "/home/amax/dakai/neuron/resnet-cats-dogs2/checkpoint-4100",
        )
        self.audio_best_path = kwargs.get(
            "audio_best_path",
            "/home/amax/dakai/neuron/wav2vec2-cats-dogs/checkpoint-525",
        )

    def _generate_run_name(self):
        """根据当前配置生成唯一的实验名称"""
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

        model_name_map = {
            "MultimodalModelDrop": "Drop",
            "MultiModalAttnModel": "Attn",
            "MultiModalAttnCLSModel": "AttnCLS",
            "HumanLikeMultimodalModel": "Human",
        }
        short_name = model_name_map.get(self.model_type, "Model")

        lr_str = f"{self.lr:.0e}"
        mp = self.model_params

        run_name = f"{short_name}_lr{lr_str}_bs{self.batch_size}"

        if "Drop" in short_name:
            run_name += f"_mask{mp['emb_mask_prob']}_v{mp['vision_drop_prob']}_a{mp['audio_drop_prob']}"
        elif "Attn" in short_name:
            run_name += f"_head{mp['attn_heads']}_dr{mp['attn_dropout']}"

        run_name += f"_{timestamp}"
        return run_name

    def create_directories(self):
        """创建必要的文件夹并保存配置"""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        config_path = os.path.join(self.ckpt_dir, "config.json")
        with open(config_path, "w") as f:
            config_dict = {
                k: v for k, v in self.__dict__.items() if not k.startswith("_")
            }
            json.dump(config_dict, f, indent=4)

        print(f"配置已保存至: {config_path}")
        print(f"模型将保存至: {self.save_path}")

    def __str__(self):
        return str(self.__dict__)

    @classmethod
    def read_json(cls, json_path, eval_mode=False):
        """
        从 JSON 文件初始化配置。

        :param json_path: config.json 的路径
        :param eval_mode:
            - False (默认): 读取参数，但生成新的 run_name 和 timestamp（用于开启新的一轮训练）。
            - True: 强制恢复 JSON 中的 run_name 和路径（用于推理测试或断点续训）。
        :return: Config 实例
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到配置文件: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        config = cls(**config_dict)

        if eval_mode:
            if "run_name" in config_dict:
                config.run_name = config_dict["run_name"]

            config.ckpt_dir = os.path.join(config.base_ckpt_dir, config.run_name)
            config.log_dir = os.path.join(config.ckpt_dir, "logs")
            config.save_path = os.path.join(config.ckpt_dir, "best_model.pth")

        return config
