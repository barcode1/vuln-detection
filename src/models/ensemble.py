import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from src.models.classifier import CodeBERTVulnClassifier
from src.models.anomaly_detector import ZeroDayAnomalyDetector
import numpy as np

class EnsembleVulnDetector(nn.Module):
    """
    Ensemble مدل سوپروایزد و آنومالی‌دیتکشن
    برای تشخیص هم حملات شناخته‌شده و هم zero-day
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # شاخه اصلی طبقه‌بندی
        self.classifier = CodeBERTVulnClassifier(config)

        # شاخه آنومالی‌دیتکشن
        self.anomaly_detector = ZeroDayAnomalyDetector(config['anomaly_detection'])

        # Fusion weights (یادگیرنده)
        self.fusion_weights = nn.Parameter(torch.tensor([0.6, 0.4], dtype=torch.float32))

        # Sigmoid for anomaly score
        self.sigmoid = nn.Sigmoid()

        # Threshold for anomaly flag
        self.anomaly_threshold = config.get('anomaly_threshold', 0.5)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                word2vec_embeds: torch.Tensor,
                fasttext_embeds: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass کامل
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            word2vec_embeds: [batch, 300]
            fasttext_embeds: [batch, 300]
            labels: Optional [batch]
        """
        # 1. شاخه طبقه‌بندی سوپروایزد
        classifier_output = self.classifier(
            input_ids, attention_mask, word2vec_embeds, fasttext_embeds, labels
        )

        classification_logits = classifier_output['logits']

        # 2. شاخه آنومالی (فقط در حالت ارزیابی)
        anomaly_scores = torch.zeros((input_ids.size(0), 1)).to(input_ids.device)

        if not self.training:
            # استخراج ویژگی از Sec-BERT
            with torch.no_grad():
                features = classifier_output['features']

            # تشخیص آنومالی
            anomaly_results = self.anomaly_detector.detect(
                features.cpu().numpy()
            )
            anomaly_scores = torch.FloatTensor(anomaly_results.reshape(-1, 1)).to(input_ids.device)

        # 3. Fusion با وزن‌های یادگیرنده
        weights = torch.softmax(self.fusion_weights, dim=0)

        # ترکیب احتمالات
        probs = F.softmax(classification_logits, dim=1)

        # افزایش وزن کلاس Unknown/Anomaly در صورت تشخیص آنومالی
        if anomaly_scores.sum() > 0:
            # اینجا فرض می‌کنیم کلاس آخر (index 3) Unknown/Anomaly است
            probs[:, 3] = torch.clamp(
                probs[:, 3] + anomaly_scores.squeeze() * 0.4,
                max=1.0
            )

        # خروجی نهایی
        final_logits = torch.log(probs + 1e-8)

        outputs = {
            'logits': final_logits,
            'classification_logits': classification_logits,
            'anomaly_scores': anomaly_scores,
            'fusion_weights': weights,
            'loss': classifier_output.get('loss', None)
        }

        return outputs

    def train_anomaly_detector(self, normal_features: np.ndarray):
        """آموزش آنومالی‌دیتکشن با داده‌های نرمال"""
        self.anomaly_detector.train(normal_features)

    def set_anomaly_threshold(self, threshold: float):
        """تنظیم threshold برای تشخیص آنومالی"""
        self.anomaly_threshold = threshold