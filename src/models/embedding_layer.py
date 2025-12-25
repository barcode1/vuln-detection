import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Tuple
import sys
import os

# اضافه کردن مسیر برای import
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.models.vulnerability_keywords import VULNERABILITY_KEYWORDS


class MultiEmbeddingFusion(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # Sec-BERT Encoder
        self.sec_bert = AutoModel.from_pretrained(
            config['sec_bert']['model_name'],
            output_hidden_states=True
        )

        # Tokenizer برای استخراج IDs
        self.tokenizer = AutoTokenizer.from_pretrained(config['sec_bert']['model_name'])

        # استخراج token IDs کلمات کلیدی
        self.keyword_ids = self._extract_keyword_ids()

        # Projection layers
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

        # وزن‌های یادگیرنده برای ترکیب جاسازی‌ها
        self.embedding_weights = nn.Parameter(torch.ones(3) / 3.0)

        # Keyword weighting (یادگیرنده)
        self.keyword_weights = nn.Parameter(torch.ones(768) * 2.0)  # وزن اولیه 2x

        # Layer normalization
        self.layer_norm = nn.LayerNorm(768)

        # Freeze early layers
        self._freeze_layers(config.get('freeze_layers', 8))

    def _freeze_layers(self, num_layers: int):
        for param in self.sec_bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.sec_bert.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def _extract_keyword_ids(self) -> torch.Tensor:
        """استخراج token IDs تمام کلمات کلیدی"""
        all_keywords = []
        for category in VULNERABILITY_KEYWORDS.values():
            all_keywords.extend(category)

        # Tokenize هر کلمه
        token_ids = []
        for word in all_keywords:
            # هر کلمه ممکن چند توکن داشته باشد
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            token_ids.extend(tokens)

        # حذف تکراری‌ها
        unique_ids = list(set(token_ids))

        return torch.tensor(unique_ids, dtype=torch.long)

    def _apply_keyword_weighting(self, embeddings: torch.Tensor,
                                 input_ids: torch.Tensor) -> torch.Tensor:
        """
        وزن‌دهی به کلمات کلیدی مخرب
        embeddings: [batch, seq_len, 768]
        input_ids: [batch, seq_len]
        """
        # 1. ماسک boolean: کجا کلمه کلیدی است
        mask = torch.isin(input_ids, self.keyword_ids.to(input_ids.device))
        # mask: [batch, seq_len]

        # 2. گستراندن ماسک به shape embeddings
        mask_expanded = mask.unsqueeze(-1).expand_as(embeddings)
        # mask_expanded: [batch, seq_len, 768]

        # 3. وزن پایه (۱) برای همه جا
        weights = torch.ones_like(embeddings)

        # 4. وزن کلیدی (یادگیرنده) و broadcast
        weight_factor = self.keyword_weights.sigmoid() * 2.0  # [768]
        weight_factor_broadcasted = weight_factor.view(1, 1, -1).expand_as(embeddings)
        # weight_factor_broadcasted: [batch, seq_len, 768]

        # 5. جایی که mask=True، وزن کلیدی را اعمال کن
        weights = torch.where(mask_expanded, weight_factor_broadcasted, weights)
        # where(condition, value_if_true, value_if_false)

        # 6. ضرب در embeddings
        return embeddings * weights

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                word2vec_embeds: torch.Tensor, fasttext_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Sec-BERT encoding
        bert_outputs = self.sec_bert(input_ids=input_ids, attention_mask=attention_mask)
        sec_bert_embeds = bert_outputs.last_hidden_state

        # 2. Projection Word2Vec و FastText
        batch_size, seq_len, _ = sec_bert_embeds.size()
        w2v_proj = self.word2vec_proj(word2vec_embeds).unsqueeze(1).repeat(1, seq_len, 1)
        ft_proj = self.fasttext_proj(fasttext_embeds).unsqueeze(1).repeat(1, seq_len, 1)

        # 3. Stack: [batch, seq_len, 3, 768]
        stack = torch.stack([sec_bert_embeds, w2v_proj, ft_proj], dim=2)

        # 4. Reshape: [batch*seq_len, 3, 768]
        stack_flat = stack.view(batch_size * seq_len, 3, 768)

        # 5. Attention
        fused_flat, _ = self.fusion_attention(stack_flat, stack_flat, stack_flat)

        # 6. Combine: [batch*seq_len, 768]
        weights = torch.softmax(self.embedding_weights, dim=0)
        fused_weighted = fused_flat * weights.view(1, 3, 1)
        fused_summed = torch.sum(fused_weighted, dim=1)

        # 7. Reshape: [batch, seq_len, 768]
        fused = fused_summed.view(batch_size, seq_len, 768)

        # 8. Keyword weighting
        fused = self._apply_keyword_weighting(fused, input_ids)

        # 9. Layer normalization
        return self.layer_norm(fused), sec_bert_embeds