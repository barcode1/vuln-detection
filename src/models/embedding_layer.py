# import torch
# import torch.nn as nn
# from transformers import AutoModel
# import torch.nn.functional as F
# from typing import Dict, Any, Tuple
#
#
# class MultiEmbeddingFusion(nn.Module):
#     """
#     فقط تولید جاسازی‌های ترکیبی (بدون طبقه‌بندی)
#     """
#
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__()
#
#         # Sec-BERT فقط برای encoding
#         self.sec_bert = AutoModel.from_pretrained(
#             config['sec_bert']['model_name'],
#             output_hidden_states=True
#         )
#
#         # Projection layers
#         self.word2vec_proj = nn.Sequential(
#             nn.Linear(300, 768),
#             nn.ReLU(),
#             nn.Dropout(config.get('embedding_dropout', 0.2))
#         )
#
#         self.fasttext_proj = nn.Sequential(
#             nn.Linear(300, 768),
#             nn.ReLU(),
#             nn.Dropout(config.get('embedding_dropout', 0.2))
#         )
#
#         # Fusion attention
#         self.fusion_attention = nn.MultiheadAttention(
#             embed_dim=768,
#             num_heads=8,
#             dropout=config.get('attention_dropout', 0.2),
#             batch_first=True
#         )
#         self.embedding_weights = nn.Parameter(torch.ones(3) / 3.0)
#         # Keyword weighting
#         self.keyword_weights = nn.Parameter(torch.ones(768) * 1.5)
#         self.layer_norm = nn.LayerNorm(768)
#
#         # Freeze early layers
#         self._freeze_layers(config.get('freeze_layers', 8))
#
#     def _freeze_layers(self, num_layers: int):
#         for param in self.sec_bert.embeddings.parameters():
#             param.requires_grad = False
#         for layer in self.sec_bert.encoder.layer[:num_layers]:
#             for param in layer.parameters():
#                 param.requires_grad = False
#
#     def _apply_keyword_weighting(self, embeddings: torch.Tensor,
#                                  input_ids: torch.Tensor) -> torch.Tensor:
#         """وزن‌دهی به کلمات کلیدی"""
#         mask = torch.isin(input_ids, torch.tensor([...]))  # token IDs کلیدی
#         weights = torch.ones_like(embeddings)
#         weights[mask.unsqueeze(-1).expand_as(embeddings)] = self.keyword_weights.sigmoid() * 2.0
#         return embeddings * weights
#
#     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
#                 word2vec_embeds: torch.Tensor, fasttext_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Returns:
#             fused_embeddings: [batch, seq_len, 768]
#             sec_bert_hidden: [batch, seq_len, 768] (برای CNN-BiLSTM)
#         """
#         # Sec-BERT encoding
#         bert_outputs = self.sec_bert(input_ids=input_ids, attention_mask=attention_mask)
#         sec_bert_embeds = bert_outputs.last_hidden_state
#
#         # Projection W2V/FT
#         batch_size, seq_len, _ = sec_bert_embeds.size()
#         w2v_proj = self.word2vec_proj(word2vec_embeds).unsqueeze(1).repeat(1, seq_len, 1)
#         ft_proj = self.fasttext_proj(fasttext_embeds).unsqueeze(1).repeat(1, seq_len, 1)
#
#         # Fusion
#         stack = torch.stack([sec_bert_embeds, w2v_proj, ft_proj], dim=2)
#         stack_flat = stack.view(batch_size * seq_len, 3, 768)
#         # fused_flat, _ = self.fusion_attention(stack_flat, stack_flat, stack_flat)
#         # fused = fused_flat.view(batch_size, seq_len, 768)
#         fused_flat, _ = self.fusion_attention(stack_flat, stack_flat, stack_flat)
#         # fused_flat: [batch*seq_len, 3, 768]
#
#         weights = torch.softmax(self.embedding_weights, dim=0)  # [3]
#         fused_weighted = fused_flat * weights.view(1, 3, 1)  # [batch*seq_len, 3, 768]
#         fused_summed = torch.sum(fused_weighted, dim=1)  # [batch*seq_len, 768]
#         fused = fused_summed.view(batch_size, seq_len, 768)  # [batch, seq_len, 768]
#
#         # Keyword weighting
#         weighted = self._apply_keyword_weighting(fused, input_ids)
#
#         return self.layer_norm(weighted), sec_bert_embeds
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
        """
        # ماسک کلمات کلیدی
        # input_ids: [batch, seq_len]
        # self.keyword_ids: [num_keywords]

        mask = torch.isin(input_ids, self.keyword_ids.to(input_ids.device))
        # mask: [batch, seq_len] (True/False)

        # افزودن یک بعد برای embeddings
        weights = torch.ones_like(embeddings)  # [batch, seq_len, 768]

        # جایی که mask=True، وزن بیشتر
        weight_factor = self.keyword_weights.sigmoid() * 2.0  # [768]

        # broadcast mask به تمام dimensions
        mask_expanded = mask.unsqueeze(-1).expand_as(embeddings)  # [batch, seq_len, 768]

        # اعمال وزن
        weights[mask_expanded] = weight_factor[mask_expanded]

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