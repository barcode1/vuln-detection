import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Any, Tuple


class MultiEmbeddingFusion(nn.Module):
    """
    ترکیب Word2Vec + FastText + Sec-BERT با Attention Fusion
    و وزن‌دهی کلمات کلیدی
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # Projection layers برای Word2Vec و FastText
        self.word2vec_proj = nn.Sequential(
            nn.Linear(300, 768),
            nn.ReLU(),
            nn.Dropout(config.get('embedding_dropout', 0.2))
        )

        self.fasttext_proj = nn.Sequential(
            nn.Linear(300, 768),
            nn.ReLU(),
            nn.Dropout(config.get('embedding_dropout', 0.2))
        )

        # Fusion attention
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            dropout=config.get('attention_dropout', 0.2),
            batch_first=True
        )

        # Keyword weighting (یادگیرنده)
        self.keyword_weights = nn.Parameter(torch.ones(768) * 1.5)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(768)

        # Sec-BERT model
        self.sec_bert = AutoModel.from_pretrained(
            config['sec_bert']['model_name'],
            output_hidden_states=True
        )

        # Freeze early layers
        self._freeze_layers(config.get('freeze_layers', 8))

    def _freeze_layers(self, num_layers: int):
        """فریز کردن لایه‌های اولیه Sec-BERT"""
        for param in self.sec_bert.embeddings.parameters():
            param.requires_grad = False

        for layer in self.sec_bert.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def _apply_keyword_weighting(self, embeddings: torch.Tensor,
                                 input_ids: torch.Tensor) -> torch.Tensor:
        """وزن‌دهی به کلمات کلیدی مخرب (SQL, XSS, CMDi)"""
        # ماسک کلمات کلیدی (در preprocess می‌توان token IDs را ذخیره کرد)
        mask = torch.isin(input_ids, torch.tensor([...]))  # باید token IDs کلیدی پر شود

        weights = torch.ones_like(embeddings)
        weights[mask.unsqueeze(-1).expand_as(embeddings)] = self.keyword_weights.sigmoid() * 2.0

        return embeddings * weights

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                word2vec_embeds: torch.Tensor, fasttext_embeds: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            word2vec_embeds: [batch, 300]
            fasttext_embeds: [batch, 300]
        Returns:
            fused_embeddings: [batch, seq_len, 768]
            sec_bert_embeddings: [batch, seq_len, 768]
            cls_token: [batch, 768]
        """
        # 1. Sec-BERT encoding
        bert_outputs = self.sec_bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sec_bert_embeds = bert_outputs.last_hidden_state
        cls_token = bert_outputs.pooler_output

        # 2. Projection Word2Vec و FastText
        batch_size, seq_len, _ = sec_bert_embeds.size()
        w2v_proj = self.word2vec_proj(word2vec_embeds).unsqueeze(1).repeat(1, seq_len, 1)
        ft_proj = self.fasttext_proj(fasttext_embeds).unsqueeze(1).repeat(1, seq_len, 1)

        # 3. Stack و Fusion با Attention
        stack = torch.stack([sec_bert_embeds, w2v_proj, ft_proj], dim=2)
        stack_flat = stack.view(batch_size * seq_len, 3, 768)

        fused_flat, _ = self.fusion_attention(stack_flat, stack_flat, stack_flat)
        fused = fused_flat.view(batch_size, seq_len, 768)

        # 4. وزن‌دهی کلمات کلیدی
        weighted = self._apply_keyword_weighting(fused, input_ids)

        # 5. Layer normalization
        return self.layer_norm(weighted), sec_bert_embeds, cls_token