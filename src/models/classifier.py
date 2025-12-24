import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, Any, Optional
from src.models.embedding_layer import MultiEmbeddingFusion
from src.models.feature_extractor import CNNBiLSTMFeatureExtractor
import numpy as np

class SecBERTVulnClassifier(nn.Module):
    """
    طبقه‌بندی آسیب‌پذیری‌های تزریقی با استفاده از ترکیب:
    - Word2Vec + FastText + Sec-BERT (MultiEmbeddingFusion)
    - CNN-BiLSTM Feature Extractor
    - طبقه‌بندی نهایی با Sec-BERT
    - Focal Loss برای عدم تعادل کلاس‌ها
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.num_labels = config['num_labels']

        # ترکیب جاسازی‌های معنایی (Word2Vec + FastText + Sec-BERT)
        self.embedding_fusion = MultiEmbeddingFusion(config)

        # استخراج ویژگی‌های پیشرفته (CNN-BiLSTM)
        self.feature_extractor = CNNBiLSTMFeatureExtractor(config)

        # طبقه‌بندی نهایی
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 768),  # CLS + CNN-BiLSTM features
            nn.ReLU(),
            nn.Dropout(config.get('classifier_dropout', 0.3)),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(config.get('classifier_dropout', 0.2)),
            nn.Linear(512, self.num_labels)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(768 * 2)

        # Loss function: Focal Loss
        self.loss_fn = FocalLoss(
            alpha=config.get('focal_alpha', [0.25, 0.25, 0.25, 0.25]),
            gamma=config.get('focal_gamma', 2.0),
            num_classes=self.num_labels
        )

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                word2vec_embeds: torch.Tensor,
                fasttext_embeds: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass کامل
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            word2vec_embeds: [batch_size, 300]
            fasttext_embeds: [batch_size, 300]
            labels: Optional [batch_size]
        """
        # 1. ترکیب جاسازی‌ها + وزن‌دهی کلمات کلیدی
        fused_embeds, sec_bert_embeds, cls_token = self.embedding_fusion(
            input_ids, attention_mask, word2vec_embeds, fasttext_embeds
        )

        # 2. استخراج ویژگی‌های پیچیده
        cnn_bilstm_features, attn_weights = self.feature_extractor(sec_bert_embeds)

        # 3. ترکیب با CLS token Sec-BERT
        combined = torch.cat([cls_token, cnn_bilstm_features], dim=-1)
        combined = self.layer_norm(combined)

        # 4. طبقه‌بندی نهایی
        logits = self.classifier(combined)

        outputs = {
            'logits': logits,
            'features': cnn_bilstm_features,  # برای آنومالی‌دیتکشن
            'attention_weights': attn_weights,
            'fused_embeddings': fused_embeds
        }

        # 5. محاسبه loss
        if labels is not None:
            outputs['loss'] = self.loss_fn(logits, labels)

        return outputs


class FocalLoss(nn.Module):
    """
    Focal Loss برای مقابله با عدم تعادل کلاس‌ها
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = 'mean', num_classes: int = 4):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        # α می‌تواند برای هر کلاس متفاوت باشد
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = torch.tensor([alpha] * num_classes, dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_one_hot = F.one_hot(targets, num_classes=self.alpha.size(0)).float()
        probs = F.softmax(logits, dim=1)
        pt = torch.sum(targets_one_hot * probs, dim=1)
        alpha_t = torch.sum(targets_one_hot * self.alpha.to(logits.device), dim=1)
        focal_factor = (1 - pt + 1e-8) ** self.gamma

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        focal_loss = alpha_t * focal_factor * ce_loss

        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()