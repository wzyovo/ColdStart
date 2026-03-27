from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for src/coldstart/model.py. Install dependencies from requirements.txt first."
    ) from exc


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(token_ids)
        conv_input = embedded.transpose(1, 2)
        conv_output = torch.relu(self.conv(conv_input)).transpose(1, 2)
        seq_output, _ = self.lstm(conv_output)
        seq_output = self.dropout(seq_output)
        # 通过注意力池化突出文本里更关键的口味和场景词。
        attn_logits = self.attn(seq_output).squeeze(-1)
        padding_mask = token_ids.eq(0)
        attn_logits = attn_logits.masked_fill(padding_mask, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(seq_output * attn_weights.unsqueeze(-1), dim=1)
        return seq_output, pooled


class InteractionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, interaction_seq: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(interaction_seq)
        return self.dropout(output[:, -1, :])


class AttentionFusion(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, fusion_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(query_dim, fusion_dim)
        self.key_proj = nn.Linear(key_dim, fusion_dim)
        self.value_proj = nn.Linear(key_dim, fusion_dim)
        self.scale = fusion_dim ** -0.5

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_proj(query).unsqueeze(1)
        k = self.key_proj(memory)
        v = self.value_proj(memory)
        attn_scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        fused = torch.matmul(attn_weights, v).squeeze(1)
        return fused, attn_weights.squeeze(1)


class MultiModalColdStartModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        text_embed_dim: int,
        text_hidden_dim: int,
        image_dim: int,
        context_dim: int,
        tag_dim: int,
        lstm_hidden_dim: int,
        fusion_dim: int,
        label_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, text_embed_dim, text_hidden_dim, dropout)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.context_proj = nn.Linear(context_dim, fusion_dim)
        self.tag_embedding = nn.Embedding(256, tag_dim, padding_idx=0)
        self.tag_proj = nn.Linear(tag_dim, fusion_dim)
        self.interaction_encoder = InteractionEncoder(input_dim=4, hidden_dim=lstm_hidden_dim, dropout=dropout)
        self.user_proj = nn.Sequential(
            nn.Linear(fusion_dim + lstm_hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.item_proj = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion = AttentionFusion(query_dim=fusion_dim, key_dim=fusion_dim, fusion_dim=fusion_dim)
        self.rank_head = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )
        self.multilabel_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, label_count),
        )

    def forward(
        self,
        user_context: torch.Tensor,
        interaction_seq: torch.Tensor,
        item_text_tokens: torch.Tensor,
        item_image_vectors: torch.Tensor,
        item_tag_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_seq, text_repr = self.text_encoder(item_text_tokens)
        image_repr = self.image_proj(item_image_vectors)
        tag_repr = self.tag_proj(self.tag_embedding(item_tag_ids).mean(dim=1))
        item_memory = torch.stack([text_repr, image_repr, tag_repr], dim=1)
        # 同时保留模态级表示和融合后的商品摘要，分别用于注意力和最终排序。
        item_summary = self.item_proj(torch.cat([text_repr, image_repr, tag_repr], dim=1))

        interaction_repr = self.interaction_encoder(interaction_seq)
        context_repr = self.context_proj(user_context)
        user_query = self.user_proj(torch.cat([context_repr, interaction_repr], dim=1))

        # 用用户表示去关注商品的文本、图像和标签三种视图。
        fused_item, attention_weights = self.fusion(user_query, item_memory)
        joint = torch.cat([user_query, fused_item, item_summary], dim=1)
        ranking_score = self.rank_head(joint).squeeze(1)
        multilabel_logits = self.multilabel_head(user_query)

        return {
            "ranking_score": ranking_score,
            "multilabel_logits": multilabel_logits,
            "attention_weights": attention_weights,
            "text_token_features": text_seq,
        }
