import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoConfig
from typing import Dict, Any, Optional, Tuple
from src.models.embedding_layer import MultiEmbeddingFusion
from src.models.feature_extractor import CNNBiLSTMFeatureExtractor
class CodeBERTVulnClassifier(nn.Module):
    """
    طبقه‌بندی نهایی با CodeBERT (فاین‌تون‌شده)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.num_labels = config['num_labels']

        # 1. ترکیب جاسازی‌ها (بدون طبقه‌بندی)
        self.embedding_fusion = MultiEmbeddingFusion(config)

        # 2. استخراج ویژگی‌های پیشرفته
        self.feature_extractor = CNNBiLSTMFeatureExtractor(config)

        # 3. CodeBERT برای طبقه‌بندی نهایی
        self.codebert = AutoModel.from_pretrained(
            config['classification']['model_name'],
            output_hidden_states=True
        )
        cls_dim = 768  # CodeBERT hidden size
        lstm_dim = config['cnn_bilstm']['lstm']['units'] * 2
        combined_dim = cls_dim + lstm_dim

        # فریز کردن بخشی از CodeBERT
        self._freeze_codebert_layers(config['classification'].get('freeze_layers', 4))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 768),  # CLS + CNN-BiLSTM features
            nn.ReLU(),
            nn.Dropout(config.get('classifier_dropout', 0.3)),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(config.get('classifier_dropout', 0.2)),
            nn.Linear(512, self.num_labels)
        )

        # Loss
        self.loss_fn = FocalLoss(
            alpha=config.get('focal_alpha', [0.216, 0.221, 0.310, 0.253]),
            gamma=config.get('focal_gamma', 2.0),
            num_classes=self.num_labels
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(combined_dim)

    def _freeze_codebert_layers(self, num_layers: int):
        """فریز کردن لایه‌های اولیه CodeBERT"""
        for param in self.codebert.embeddings.parameters():
            param.requires_grad = False

        for layer in self.codebert.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                word2vec_embeds: torch.Tensor,
                fasttext_embeds: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        # 1. ترکیب جاسازی‌ها + وزن‌دهی
        fused_embeds, sec_bert_hidden = self.embedding_fusion(
            input_ids, attention_mask, word2vec_embeds, fasttext_embeds
        )

        # 2. استخراج ویژگی‌های پیشرفته
        cnn_bilstm_features, attn_weights = self.feature_extractor(sec_bert_hidden)

        # 3. CodeBERT برای درک معنای کد

        codebert_outputs = self.codebert(inputs_embeds=fused_embeds, attention_mask=attention_mask)
        codebert_cls = codebert_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # 4. ترکیب ویژگی‌ها
        combined = torch.cat([codebert_cls, cnn_bilstm_features], dim=-1)
        combined = self.layer_norm(combined)

        # 5. طبقه‌بندی نهایی
        logits = self.classifier(combined)

        outputs = {
            'logits': logits,
            'features': cnn_bilstm_features,
            'attention_weights': attn_weights,
            'codebert_hidden': codebert_cls
        }

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