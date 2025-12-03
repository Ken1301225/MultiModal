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
    
    
class MultiModalAttnModel(nn.Module):
    def __init__(
        self,
        vision_model,
        audio_model,
        shared_dim=512,
        num_classes=2,
        attn_heads=8,  # 注意力头数
        attn_dropout=0.2,  # 注意力权重 dropout
        linear_dropout=0.2,  # 全连接层 dropout
        vision_drop_prob=0.0,  # 整路视觉 drop 概率
        audio_drop_prob=0.0,  # 整路音频 drop 概
    ):
        super().__init__()

        # --- 1. 编码器和投影层 (与之前模型相同) ---
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

        # --- 2. 多模态融合层 (使用自注意力) ---
        # 我们将 v_emb 和 a_emb 看作一个长度为 2 的序列
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,  # 输入形状为 (batch, seq_len, dim)
        )
        # LayerNorm 通常与 Attention 配合使用，增加稳定性
        self.fusion_norm = nn.LayerNorm(shared_dim)

        # --- 3. 分类器 ---
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
        # mask 形状为 (batch_size, seq_len)，在我们的例子中是 (bsz, 2)
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

        # --- 防止一个样本的两个模态都被屏蔽 ---
        # 1. 找到哪些样本的两个模态都被屏蔽了
        all_masked_indices = mask.all(dim=1)
        
        # 2. 如果存在这样的样本
        if all_masked_indices.any():
            # 3. 为这些被双重屏蔽的样本，生成一个随机的索引（0或1）来恢复
            # torch.randint(0, 2, ...) 会生成 0 或 1 的随机整数
            num_all_masked = all_masked_indices.sum()
            indices_to_unmask = torch.randint(0, 2, (num_all_masked,), device=device)
            
            # 4. 将这些样本的对应随机索引位置的 mask 设置为 False
            mask[all_masked_indices, indices_to_unmask] = False

        return mask

    def forward(self, pixel_values, input_values, labels=None):
        # --- 1. 特征提取与投影 ---
        v_out = self.vision_encoder(pixel_values)
        v_feat = v_out.pooler_output.flatten(1)
        v_emb = self.vision_proj(v_feat)  # (batch, shared_dim)

        a_out = self.audio_encoder(input_values)
        a_feat = torch.mean(a_out.last_hidden_state, dim=1)
        a_emb = self.audio_proj(a_feat)  # (batch, shared_dim)

        # --- 2. 自注意力融合 ---
        # 将 v_emb 和 a_emb 视为一个序列，长度为 2
        # v_emb: (batch, shared_dim) -> (batch, 1, shared_dim)
        # a_emb: (batch, shared_dim) -> (batch, 1, shared_dim)
        # multi_modal_seq: (batch, 2, shared_dim)
        multi_modal_seq = torch.stack([v_emb, a_emb], dim=1)
        
        # --- 创建 key_padding_mask ---
        key_padding_mask = None
        if self.training and (self.vision_drop_prob > 0 or self.audio_drop_prob > 0):
            bsz, _, _ = multi_modal_seq.shape
            key_padding_mask = self._create_key_padding_mask(bsz, multi_modal_seq.device)

        # 自注意力计算
        # attn_output: (batch, 2, shared_dim)
        attn_output, _ = self.fusion_attention(
            query=multi_modal_seq,
            key=multi_modal_seq,
            value=multi_modal_seq,
            key_padding_mask=key_padding_mask # 训练时随机 mask 掉某个模态
        )
        
        # 残差连接和层归一化
        fused_seq = self.fusion_norm(multi_modal_seq + attn_output)

        # --- 3. 融合与决策 ---
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

        # --- 2. 可学习的 [CLS] Token ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, shared_dim))

        # --- 3. 多模态融合层 (自注意力) ---
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(shared_dim)

        # --- 4. 分类器 (基于 [CLS] Token) ---
        self.classifier = nn.Linear(shared_dim, num_classes)

    def _create_key_padding_mask(self, bsz, device):
        """
        为 [CLS, vision, audio] 序列生成 key_padding_mask。
        只在 self.training=True 时调用。
        """
        # mask 形状为 (bsz, 3)，True 表示忽略该 key
        mask = torch.zeros(bsz, 3, dtype=torch.bool, device=device)

        if self.vision_drop_prob > 0.0:
            vision_drop_indices = torch.rand(bsz, device=device) < self.vision_drop_prob
            mask[vision_drop_indices, 1] = True  # 序列中第1个是视觉

        if self.audio_drop_prob > 0.0:
            audio_drop_indices = torch.rand(bsz, device=device) < self.audio_drop_prob
            mask[audio_drop_indices, 2] = True  # 序列中第2个是听觉

        # 防止视觉和听觉都被屏蔽 (CLS Token 不参与此逻辑)
        all_masked_indices = mask[:, 1:].all(dim=1)
        if all_masked_indices.any():
            num_all_masked = all_masked_indices.sum()
            # 随机恢复 1 (vision) 或 2 (audio)
            indices_to_unmask = torch.randint(1, 3, (num_all_masked,), device=device)
            mask[all_masked_indices, indices_to_unmask] = False

        return mask

    def forward(self, pixel_values, input_values, labels=None):
        # --- 1. 特征提取与投影 ---
        v_out = self.vision_encoder(pixel_values)
        v_feat = v_out.pooler_output.flatten(1)
        v_emb = self.vision_proj(v_feat)

        a_out = self.audio_encoder(input_values)
        a_feat = torch.mean(a_out.last_hidden_state, dim=1)
        a_emb = self.audio_proj(a_feat)

        # --- 2. 自注意力融合 ---
        # 2.1 准备序列，加入 [CLS] Token
        bsz = v_emb.shape[0]
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        
        # 序列变为: [CLS, Vision, Audio]
        multi_modal_seq = torch.cat((cls_tokens, v_emb.unsqueeze(1), a_emb.unsqueeze(1)), dim=1)

        # 2.2 创建 key_padding_mask
        key_padding_mask = None
        if self.training and (self.vision_drop_prob > 0 or self.audio_drop_prob > 0):
            key_padding_mask = self._create_key_padding_mask(bsz, multi_modal_seq.device)

        # 2.3 自注意力计算
        attn_output, _ = self.fusion_attention(
            query=multi_modal_seq,
            key=multi_modal_seq,
            value=multi_modal_seq,
            key_padding_mask=key_padding_mask,
        )

        # 2.4 残差连接和层归一化
        fused_seq = self.fusion_norm(multi_modal_seq + attn_output)

        # --- 3. 决策 ---
        # 只取出 [CLS] Token 对应的输出 (序列的第一个)
        cls_output = fused_seq[:, 0, :]
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}