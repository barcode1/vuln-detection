import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple


class Autoencoder(nn.Module):
    """Autoencoder برای تشخیص آنومالی بر اساس reconstruction error"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, config['latent_dim'])
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config['latent_dim'], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 768),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class ZeroDayAnomalyDetector:
    """
    تشخیص آنومالی برای zero-day exploits
    ترکیب Autoencoder + Isolation Forest
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.autoencoder = Autoencoder(config['autoencoder'])
        self.isolation_forest = IsolationForest(
            n_estimators=config['isolation_forest']['n_estimators'],
            contamination=config['isolation_forest']['contamination'],
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.reconstruction_threshold = None

    def train(self, normal_features: np.ndarray):
        """آموزش بر روی داده‌های نرمال (benign)"""
        # 1. Autoencoder training
        optimizer = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=self.config['autoencoder'].get('learning_rate', 0.001)
        )
        criterion = nn.MSELoss()

        self.autoencoder.train()
        for epoch in range(self.config['autoencoder'].get('epochs', 50)):
            for batch in self._create_batches(normal_features):
                batch_tensor = torch.FloatTensor(batch)
                reconstructed, _ = self.autoencoder(batch_tensor)

                loss = criterion(reconstructed, batch_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 2. Isolation Forest training
        scaled_features = self.scaler.fit_transform(normal_features)
        self.isolation_forest.fit(scaled_features)

        # 3. محاسبه threshold برای reconstruction error
        self._calculate_threshold(normal_features)

    def detect(self, features: np.ndarray) -> np.ndarray:
        """تشخیص آنومالی (0=normal, 1=anomaly)"""
        # Autoencoder error
        features_tensor = torch.FloatTensor(features)
        self.autoencoder.eval()

        with torch.no_grad():
            reconstructed, _ = self.autoencoder(features_tensor)

        mse = np.mean(np.square(features - reconstructed.numpy()), axis=1)
        ae_anomalies = (mse > self.reconstruction_threshold).astype(int)

        # Isolation Forest
        scaled_features = self.scaler.transform(features)
        if_anomalies = (self.isolation_forest.predict(scaled_features) == -1).astype(int)

        # ترکیب OR
        return np.logical_or(ae_anomalies, if_anomalies).astype(int)

    def _calculate_threshold(self, normal_features: np.ndarray, percentile: int = 95):
        """محاسبه threshold بر اساس داده‌های نرمال"""
        features_tensor = torch.FloatTensor(normal_features)
        self.autoencoder.eval()

        with torch.no_grad():
            reconstructed, _ = self.autoencoder(features_tensor)

        mse = np.mean(np.square(normal_features - reconstructed.numpy()), axis=1)
        self.reconstruction_threshold = np.percentile(mse, percentile)

    def _create_batches(self, data: np.ndarray, batch_size: int = 32):
        """ساخت batch برای آموزش"""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]