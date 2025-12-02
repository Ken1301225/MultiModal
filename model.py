import torch
import torch.nn as nn
import torch.nn.functional as F


class HumanLikeMultimodalModel(nn.Module):
    def __init__(self, vision_model, audio_model, shared_dim=512, num_classes=2):
        super().__init__()

        self.vision_encoder = vision_model.resnet  # 提取 ResNet 主干
        self.vision_dim = 2048  # ResNet50 最后一层特征维度

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # 2. 音频编码器 (Wav2Vec2)
        self.audio_encoder = audio_model.wav2vec2  # 提取 Wav2Vec2 主干
        self.audio_dim = 768  # Wav2Vec2 Base 特征维度

        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        # 3. 投影层 (Projection Heads) -> 映射到相同长度 tokens
        self.vision_proj = nn.Linear(self.vision_dim, shared_dim)
        self.audio_proj = nn.Linear(self.audio_dim, shared_dim)

        self.classifier = nn.Sequential(
            nn.Linear(shared_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values, input_values, labels=None):
        # --- 1. 视觉通路 ---
        # ResNet 输出: (batch, 2048, 7, 7) -> pool -> (batch, 2048, 1, 1)
        v_out = self.vision_encoder(pixel_values)
        # 获取 pooled output (batch, 2048)
        v_feat = v_out.pooler_output.flatten(1)

        # --- 2. 听觉通路 ---
        # Wav2Vec2 输出: (batch, seq_len, 768)
        a_out = self.audio_encoder(input_values)
        # 我们取平均池化 (Mean Pooling) 得到全局特征 (batch, 768)
        a_feat = torch.mean(a_out.last_hidden_state, dim=1)

        # --- 3. 投影对齐 (Encoding to same length) ---
        v_emb = self.vision_proj(v_feat)  # (batch, 512)
        a_emb = self.audio_proj(a_feat)  # (batch, 512)

        # --- 4. 计算对比损失 (Contrastive Loss) ---
        # loss_contrastive = 0
        # if labels is not None:

        #     # 构建目标：同类样本相似度应为1，不同类为-1 (或者使用 InfoNCE)
        #     # 这里简化：只拉近同一个样本的视听距离 (Self-Supervised style)
        #     # 或者更强的：Supervised Contrastive Loss

        #     # 简单实现：Cosine Embedding Loss
        #     # 创建 target: 1 表示应该相似 (这里全是正样本对，因为我们是配对输入的)
        #     target = torch.ones(v_emb.size(0)).to(v_emb.device)
        #     loss_contrastive = F.cosine_embedding_loss(v_emb, a_emb, target)

        # --- 5. 融合与决策 ---
        combined_feat = torch.cat((v_emb, a_emb), dim=1)  # (batch, 1024)
        logits = self.classifier(combined_feat)

        loss = None
        if labels is not None:
            loss_cls = F.cross_entropy(logits, labels)
            # 总损失 = 分类损失 + lambda * 对比损失
            # lambda 系数决定了你多看重"对齐"
            # loss = loss_cls + 0.5 * loss_contrastive
            loss = loss_cls

        # return {"loss": loss, "logits": logits, "contrastive_loss": loss_contrastive}
        return {"loss": loss, "logits": logits}


def print_trainable_parameters(model):
    """
    打印模型中所有可训练的参数。
    """
    trainable_params = 0
    all_param = 0

    print("\n" + "=" * 60)
    print("可训练参数详情:")
    print("=" * 60)

    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(
                f"✓ {name:50s} | Shape: {str(list(param.shape)):20s} | Count: {param.numel():,}"
            )

    print("=" * 60)
    print(f"可训练参数总数: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"所有参数总数:   {all_param:,} ({all_param / 1e6:.2f}M)")
    print(f"可训练参数占比: {100 * trainable_params / all_param:.2f}%")
    print("=" * 60 + "\n")


class MultimodalModelDrop(nn.Module):
    def __init__(
        self,
        vision_model,
        audio_model,
        shared_dim=512,
        num_classes=2,
        vision_drop_prob=0.0,  # 新增：整路视觉 drop 概率
        audio_drop_prob=0.0,  # 新增：整路音频 drop 概率
        emb_mask_prob=0.0,  # 新增：在 embedding 维度上随机 mask 的概率（可选）
    ):
        super().__init__()

        self.vision_encoder = vision_model.resnet  # 提取 ResNet 主干
        self.vision_dim = 2048  # ResNet50 最后一层特征维度

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # 2. 音频编码器 (Wav2Vec2)
        self.audio_encoder = audio_model.wav2vec2  # 提取 Wav2Vec2 主干
        self.audio_dim = 768  # Wav2Vec2 Base 特征维度

        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        # 3. 投影层 (Projection Heads) -> 映射到相同长度 tokens
        self.vision_proj = nn.Linear(self.vision_dim, shared_dim)
        self.audio_proj = nn.Linear(self.audio_dim, shared_dim)

        self.classifier = nn.Sequential(
            nn.Linear(shared_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

        # --- 新增：mask 控制参数 ---
        self.vision_drop_prob = vision_drop_prob
        self.audio_drop_prob = audio_drop_prob
        self.emb_mask_prob = emb_mask_prob

    def _apply_random_route_drop(self, v_emb, a_emb):
        """
        随机将整个视觉 / 音频通路的 embedding 置零（类似通道级 dropout）。
        只在 self.training=True 时调用。
        """
        bsz = v_emb.size(0)
        device = v_emb.device

        # 对每个样本，独立决定是否 drop vision / audio
        if self.vision_drop_prob > 0.0:
            vision_keep_mask = (
                torch.rand(bsz, 1, device=device) > self.vision_drop_prob
            ).float()
            # 保留的样本乘 1，被 drop 的样本乘 0
            v_emb = v_emb * vision_keep_mask
        if self.audio_drop_prob > 0.0:
            audio_keep_mask = (
                torch.rand(bsz, 1, device=device) > self.audio_drop_prob
            ).float()
            a_emb = a_emb * audio_keep_mask

        return v_emb, a_emb

    def _apply_random_emb_mask(self, emb):
        """
        在 embedding 维度上随机 mask 一部分（逐维置零），类似自定义 dropout。
        emb: (batch, dim)
        """
        if self.emb_mask_prob <= 0.0 or not self.training:
            return emb
        # dropout 的标准实现也可以：nn.Dropout(p=self.emb_mask_prob)(emb)
        # 这里演示按维度生成 mask
        mask = (torch.rand_like(emb, device=emb.device) > self.emb_mask_prob).float()
        return emb * mask

    def forward(self, pixel_values, input_values, labels=None):
        # --- 1. 视觉通路 ---
        # ResNet 输出: (batch, 2048, 7, 7) -> pool -> (batch, 2048, 1, 1)
        v_out = self.vision_encoder(pixel_values)
        # 获取 pooled output (batch, 2048)
        v_feat = v_out.pooler_output.flatten(1)

        # --- 2. 听觉通路 ---
        # Wav2Vec2 输出: (batch, seq_len, 768)
        a_out = self.audio_encoder(input_values)
        # 我们取平均池化 (Mean Pooling) 得到全局特征 (batch, 768)
        a_feat = torch.mean(a_out.last_hidden_state, dim=1)

        # --- 3. 投影对齐 (Encoding to same length) ---
        v_emb = self.vision_proj(v_feat)  # (batch, shared_dim)
        a_emb = self.audio_proj(a_feat)  # (batch, shared_dim)

        # --- 4. 训练时随机 mask ---
        if self.training:
            # 4.1 随机整路 drop vision / audio
            v_emb, a_emb = self._apply_random_route_drop(v_emb, a_emb)
            # 4.2 在 embedding 内部再做一点逐维 mask（可选）
            v_emb = self._apply_random_emb_mask(v_emb)
            a_emb = self._apply_random_emb_mask(a_emb)

        # --- 5. 融合与决策 ---
        combined_feat = torch.cat((v_emb, a_emb), dim=1)  # (batch, 2 * shared_dim)
        logits = self.classifier(combined_feat)

        loss = None
        if labels is not None:
            loss_cls = F.cross_entropy(logits, labels)
            loss = loss_cls

        return {"loss": loss, "logits": logits}
