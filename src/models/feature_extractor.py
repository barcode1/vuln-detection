import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class CNNBiLSTMFeatureExtractor(nn.Module):
    """
    استخراج ویژگی‌های پیشرفته با CNN multi-scale + BiLSTM + Self-Attention
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.hidden_dim = config['cnn_bilstm']['lstm']['units']
        self.dropout_rate = config.get('dropout', 0.3)

        # CNN multi-scale
        self.conv1 = nn.Conv1d(768, 256, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(768, 512, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv1d(768, 512, kernel_size=7, padding=3, bias=False)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv_dropout = nn.Dropout(self.dropout_rate)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=256 + 512 + 512,
            hidden_size=self.hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout_rate
        )

        # Self-Attention pooling
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,
            num_heads=8,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(self.hidden_dim * 2)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: [batch_size, seq_len, 768]
        Returns:
            features: [batch_size, 512]
            attention_weights: [batch_size, seq_len]
        """
        # Transpose for CNN
        x = embeddings.transpose(1, 2)  # [batch, 768, seq_len]

        # Multi-scale CNN
        x1 = self.conv_dropout(F.relu(self.bn1(self.conv1(x))))  # [batch, 256, seq_len]
        x2 = self.conv_dropout(F.relu(self.bn2(self.conv2(x))))  # [batch, 512, seq_len]
        x3 = self.conv_dropout(F.relu(self.bn3(self.conv3(x))))  # [batch, 512, seq_len]

        # Concatenate and transpose
        x = torch.cat([x1, x2, x3], dim=1).transpose(1, 2)  # [batch, seq_len, 1280]

        # BiLSTM
        lstm_out, _ = self.bilstm(x)  # [batch, seq_len, 512]

        # Self-Attention pooling
        attn_out, weights = self.self_attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out)

        # Global average pooling
        features = torch.mean(attn_out, dim=1)  # [batch, 512]

        return features, weights