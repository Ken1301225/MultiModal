import torch
import torch.nn as nn
import torch.nn.functional as F


class HumanLikeMultimodalModel(nn.Module):
    def __init__(self, vision_model, audio_model, shared_dim=512, num_classes=2):
        super().__init__()

        self.vision_encoder = vision_model.resnet
        self.vision_dim = 2048

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.audio_encoder = audio_model.wav2vec2
        self.audio_dim = 768

        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        self.vision_proj = nn.Linear(self.vision_dim, shared_dim)
        self.audio_proj = nn.Linear(self.audio_dim, shared_dim)

        self.classifier = nn.Sequential(
            nn.Linear(shared_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values, input_values, labels=None):
        v_out = self.vision_encoder(pixel_values)
        v_feat = v_out.pooler_output.flatten(1)

        a_out = self.audio_encoder(input_values)
        a_feat = torch.mean(a_out.last_hidden_state, dim=1)

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

        combined_feat = torch.cat((v_emb, a_emb), dim=1)  # (batch, 1024)
        logits = self.classifier(combined_feat)

        loss = None
        if labels is not None:
            loss_cls = F.cross_entropy(logits, labels)
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
        vision_drop_prob=0.0,
        audio_drop_prob=0.0,
        emb_mask_prob=0.0,
    ):
        super().__init__()

        self.vision_encoder = vision_model.resnet
        self.vision_dim = 2048

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.audio_encoder = audio_model.wav2vec2
        self.audio_dim = 768

        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        self.vision_proj = nn.Linear(self.vision_dim, shared_dim)
        self.audio_proj = nn.Linear(self.audio_dim, shared_dim)

        self.classifier = nn.Sequential(
            nn.Linear(shared_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

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

        if self.vision_drop_prob > 0.0:
            vision_keep_mask = (
                torch.rand(bsz, 1, device=device) > self.vision_drop_prob
            ).float()
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

        mask = (torch.rand_like(emb, device=emb.device) > self.emb_mask_prob).float()
        return emb * mask

    def forward(self, pixel_values, input_values, labels=None):
        #  (batch, 2048, 7, 7) -> pool -> (batch, 2048, 1, 1)
        v_out = self.vision_encoder(pixel_values)
        #  pooled output (batch, 2048)
        v_feat = v_out.pooler_output.flatten(1)

        # (batch, seq_len, 768)
        a_out = self.audio_encoder(input_values)
        # Mean Pooling) (batch, 768)
        a_feat = torch.mean(a_out.last_hidden_state, dim=1)

        v_emb = self.vision_proj(v_feat)  # (batch, shared_dim)
        a_emb = self.audio_proj(a_feat)  # (batch, shared_dim)

        if self.training:
            v_emb, a_emb = self._apply_random_route_drop(v_emb, a_emb)
            v_emb = self._apply_random_emb_mask(v_emb)
            a_emb = self._apply_random_emb_mask(a_emb)

        combined_feat = torch.cat((v_emb, a_emb), dim=1)  # (batch, 2 * shared_dim)
        logits = self.classifier(combined_feat)

        loss = None
        if labels is not None:
            loss_cls = F.cross_entropy(logits, labels)
            loss = loss_cls

        return {"loss": loss, "logits": logits}


class MultiModalAttnModel(nn.Module):
    def __init__(
        self,
        vision_model,
        audio_model,
        shared_dim=512,
        num_classes=2,
        attn_heads=8,
        attn_dropout=0.2,
        linear_dropout=0.2,
        vision_drop_prob=0.0,
        audio_drop_prob=0.0,
    ):
        super().__init__()

        self.vision_encoder = vision_model.resnet
        self.vision_dim = 2048
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.audio_encoder = audio_model.wav2vec2
        self.audio_dim = 768
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        self.vision_proj = nn.Linear(self.vision_dim, shared_dim)
        self.audio_proj = nn.Linear(self.audio_dim, shared_dim)

        self.vision_drop_prob = vision_drop_prob
        self.audio_drop_prob = audio_drop_prob

        # 我们将 v_emb 和 a_emb 看作一个长度为 2 的序列
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,  # 输入形状为 (batch, seq_len, dim)
        )
        self.fusion_norm = nn.LayerNorm(shared_dim)

        # 注意力融合后，我们仍然得到两个模态的特征，将它们拼接
        # self.classifier = nn.Sequential(
        #     nn.Linear(shared_dim * 2, 256),
        #     nn.ReLU(),
        #     nn.Dropout(linear_dropout),
        #     nn.Linear(256, num_classes),
        # )
        self.classifier = nn.Linear(shared_dim * 2, num_classes)

    def _create_key_padding_mask(self, bsz, device):
        """
        根据概率为批次中的每个样本生成 key_padding_mask。
        只在 self.training=True 时调用。
        """
        # True 表示该位置的 key 会被忽略
        mask = torch.zeros(bsz, 2, dtype=torch.bool, device=device)

        if self.vision_drop_prob > 0.0:
            # 为每个样本生成一个随机数，如果小于 drop 概率，则屏蔽视觉模态
            vision_drop_indices = torch.rand(bsz, device=device) < self.vision_drop_prob
            mask[vision_drop_indices, 0] = True

        if self.audio_drop_prob > 0.0:
            # 屏蔽听觉模态
            audio_drop_indices = torch.rand(bsz, device=device) < self.audio_drop_prob
            mask[audio_drop_indices, 1] = True

        all_masked_indices = mask.all(dim=1)

        if all_masked_indices.any():

            num_all_masked = all_masked_indices.sum()
            indices_to_unmask = torch.randint(0, 2, (num_all_masked,), device=device)

            # 将这些样本的对应随机索引位置的 mask 设置为 False
            mask[all_masked_indices, indices_to_unmask] = False

        return mask

    def forward(self, pixel_values, input_values, labels=None):
        v_out = self.vision_encoder(pixel_values)
        v_feat = v_out.pooler_output.flatten(1)
        v_emb = self.vision_proj(v_feat)  # (batch, shared_dim)

        a_out = self.audio_encoder(input_values)
        a_feat = torch.mean(a_out.last_hidden_state, dim=1)
        a_emb = self.audio_proj(a_feat)  # (batch, shared_dim)

        # 将 v_emb 和 a_emb 视为一个序列，长度为 2
        # v_emb: (batch, shared_dim) -> (batch, 1, shared_dim)
        # a_emb: (batch, shared_dim) -> (batch, 1, shared_dim)
        # multi_modal_seq: (batch, 2, shared_dim)
        multi_modal_seq = torch.stack([v_emb, a_emb], dim=1)

        key_padding_mask = None
        if self.training and (self.vision_drop_prob > 0 or self.audio_drop_prob > 0):
            bsz, _, _ = multi_modal_seq.shape
            key_padding_mask = self._create_key_padding_mask(
                bsz, multi_modal_seq.device
            )

        # attn_output: (batch, 2, shared_dim)
        attn_output, _ = self.fusion_attention(
            query=multi_modal_seq,
            key=multi_modal_seq,
            value=multi_modal_seq,
            key_padding_mask=key_padding_mask,
        )

        fused_seq = self.fusion_norm(multi_modal_seq + attn_output)

        # fused_seq[:, 0, :] 是融合后的视觉特征
        # fused_seq[:, 1, :] 是融合后的听觉特征
        combined_feat = fused_seq.flatten(start_dim=1)  # (batch, 2 * shared_dim)
        logits = self.classifier(combined_feat)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


class MultiModalAttnCLSModel(nn.Module):
    def __init__(
        self,
        vision_model,
        audio_model,
        shared_dim=512,
        num_classes=2,
        attn_heads=8,
        attn_dropout=0.2,
        vision_drop_prob=0.0,
        audio_drop_prob=0.0,
    ):
        super().__init__()

        # --- 1. 编码器和投影层 ---
        self.vision_encoder = vision_model.resnet
        self.vision_dim = 2048
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.audio_encoder = audio_model.wav2vec2
        self.audio_dim = 768
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        self.vision_proj = nn.Linear(self.vision_dim, shared_dim)
        self.audio_proj = nn.Linear(self.audio_dim, shared_dim)

        self.vision_drop_prob = vision_drop_prob
        self.audio_drop_prob = audio_drop_prob

        self.cls_token = nn.Parameter(torch.zeros(1, 1, shared_dim))

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(shared_dim)

        self.classifier = nn.Linear(shared_dim, num_classes)

    def _create_key_padding_mask(self, bsz, device):
        """
        为 [CLS, vision, audio] 序列生成 key_padding_mask。
        只在 self.training=True 时调用。
        """
        mask = torch.zeros(bsz, 3, dtype=torch.bool, device=device)

        if self.vision_drop_prob > 0.0:
            vision_drop_indices = torch.rand(bsz, device=device) < self.vision_drop_prob
            mask[vision_drop_indices, 1] = True

        if self.audio_drop_prob > 0.0:
            audio_drop_indices = torch.rand(bsz, device=device) < self.audio_drop_prob
            mask[audio_drop_indices, 2] = True

        all_masked_indices = mask[:, 1:].all(dim=1)
        if all_masked_indices.any():
            num_all_masked = all_masked_indices.sum()
            indices_to_unmask = torch.randint(1, 3, (num_all_masked,), device=device)
            mask[all_masked_indices, indices_to_unmask] = False

        return mask

    def forward(self, pixel_values, input_values, labels=None):
        v_out = self.vision_encoder(pixel_values)
        v_feat = v_out.pooler_output.flatten(1)
        v_emb = self.vision_proj(v_feat)

        a_out = self.audio_encoder(input_values)
        a_feat = torch.mean(a_out.last_hidden_state, dim=1)
        a_emb = self.audio_proj(a_feat)

        bsz = v_emb.shape[0]
        cls_tokens = self.cls_token.expand(bsz, -1, -1)

        multi_modal_seq = torch.cat(
            (cls_tokens, v_emb.unsqueeze(1), a_emb.unsqueeze(1)), dim=1
        )

        key_padding_mask = None
        if self.training and (self.vision_drop_prob > 0 or self.audio_drop_prob > 0):
            key_padding_mask = self._create_key_padding_mask(
                bsz, multi_modal_seq.device
            )

        attn_output, _ = self.fusion_attention(
            query=multi_modal_seq,
            key=multi_modal_seq,
            value=multi_modal_seq,
            key_padding_mask=key_padding_mask,
        )

        fused_seq = self.fusion_norm(multi_modal_seq + attn_output)

        cls_output = fused_seq[:, 0, :]
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}
