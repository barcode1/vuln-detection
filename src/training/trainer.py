# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from transformers import  get_linear_schedule_with_warmup
# from torch.optim import AdamW
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score
# import numpy as np
# from typing import Dict, Any
# import logging
#
#
# class VulnDetectionTrainer:
#     """
#     Trainer برای آموزش Ensemble مدل تشخیص آسیب‌پذیری
#     شامل دو مرحله: 1) آموزش classifier، 2) آموزش anomaly detector
#     """
#
#     def __init__(self, model, config: Dict[str, Any],
#                  train_dataset, val_dataset):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = model.to(self.device)
#         self.config = config
#
#         # DataLoaders
#         self.train_loader = DataLoader(
#             train_dataset,
#             batch_size=config['sec_bert']['batch_size'],
#             shuffle=True,
#             num_workers=2
#         )
#         self.val_loader = DataLoader(
#             val_dataset,
#             batch_size=config['sec_bert']['batch_size'],
#             shuffle=False,
#             num_workers=2
#         )
#
#         # Optimizer با lr جداگانه برای BERT و سایر لایه‌ها
#         bert_params = [p for n, p in model.classifier.named_parameters() if 'sec_bert' in n]
#         other_params = [p for n, p in model.classifier.named_parameters() if 'sec_bert' not in n]
#
#         self.optimizer = AdamW([
#             {'params': bert_params, 'lr': config['sec_bert']['learning_rate']},
#             {'params': other_params, 'lr': config['sec_bert']['learning_rate'] * 10}
#         ], weight_decay=config['sec_bert']['weight_decay'])
#
#         # Scheduler
#         total_steps = len(self.train_loader) * config['sec_bert']['epochs']
#         self.scheduler = get_linear_schedule_with_warmup(
#             self.optimizer,
#             num_warmup_steps=config['sec_bert']['warmup_steps'],
#             num_training_steps=total_steps
#         )
#
#         # Early stopping
#         self.best_f1 = 0
#         self.patience_counter = 0
#
#         # Logger
#         self.logger = logging.getLogger(__name__)
#         self.logger.setLevel(logging.INFO)
#
#     def train_epoch(self, epoch: int) -> float:
#         """آموزش یک epoch کامل"""
#         self.model.classifier.train()
#         self.model.anomaly_detector.autoencoder.train()  # فقط برای آموزش آنومالی
#         total_loss = 0
#
#         for batch_idx, batch in enumerate(self.train_loader):
#             # انتقال به device
#             input_ids = batch['input_ids'].to(self.device)
#             attention_mask = batch['attention_mask'].to(self.device)
#             word2vec = batch['word2vec_embeds'].to(self.device)
#             fasttext = batch['fasttext_embeds'].to(self.device)
#             labels = batch['labels'].to(self.device)
#
#             # Forward
#             outputs = self.model.classifier(
#                 input_ids, attention_mask, word2vec, fasttext, labels
#             )
#
#             loss = outputs['loss']
#
#             # Backward
#             self.optimizer.zero_grad()
#             loss.backward()
#
#             # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(
#                 self.model.classifier.parameters(),
#                 self.config['sec_bert']['max_grad_norm']
#             )
#
#             self.optimizer.step()
#             self.scheduler.step()
#
#             total_loss += loss.item()
#
#             # Logging هر 100 batch
#             if batch_idx % 100 == 0:
#                 self.logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
#
#         return total_loss / len(self.train_loader)
#
#     def validate(self) -> Dict[str, float]:
#         """ارزیابی روی داده‌های اعتبارسنجی"""
#         self.model.classifier.eval()
#         all_preds = []
#         all_labels = []
#         all_losses = []
#
#         with torch.no_grad():
#             for batch in self.val_loader:
#                 input_ids = batch['input_ids'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 word2vec = batch['word2vec_embeds'].to(self.device)
#                 fasttext = batch['fasttext_embeds'].to(self.device)
#                 labels = batch['labels'].to(self.device)
#
#                 outputs = self.model.classifier(
#                     input_ids, attention_mask, word2vec, fasttext, labels
#                 )
#
#                 logits = outputs['logits']
#                 preds = torch.argmax(logits, dim=1)
#
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
#                 all_losses.append(outputs['loss'].item())
#
#         # محاسبه معیارها
#         precision, recall, f1, _ = precision_recall_fscore_support(
#             all_labels, all_preds, average='weighted', zero_division=0
#         )
#         accuracy = accuracy_score(all_labels, all_preds)
#
#         return {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1': f1,
#             'loss': np.mean(all_losses)
#         }
#
#     def train(self):
#         """آموزش کامل مدل"""
#         self.logger.info("=" * 50)
#         self.logger.info("شروع آموزش مدل تشخیص آسیب‌پذیری")
#         self.logger.info("=" * 50)
#
#         for epoch in range(self.config['sec_bert']['epochs']):
#             self.logger.info(f"\n{'=' * 20} Epoch {epoch + 1}/{self.config['sec_bert']['epochs']} {'=' * 20}")
#
#             # آموزش
#             train_loss = self.train_epoch(epoch)
#             self.logger.info(f"Train Loss: {train_loss:.4f}")
#
#             # اعتبارسنجی
#             metrics = self.validate()
#             self.logger.info(f"Val Accuracy: {metrics['accuracy']:.4f}")
#             self.logger.info(f"Val F1: {metrics['f1']:.4f}")
#             self.logger.info(f"Val Loss: {metrics['loss']:.4f}")
#
#             # Early stopping
#             if metrics['f1'] > self.best_f1:
#                 self.best_f1 = metrics['f1']
#                 self.patience_counter = 0
#                 self._save_checkpoint(epoch, metrics)
#                 self.logger.info("✅ مدل بهتر ذخیره شد!")
#             else:
#                 self.patience_counter += 1
#                 self.logger.info(f"📉 Early stopping counter: {self.patience_counter}/{self.config.get('patience', 3)}")
#
#                 if self.patience_counter >= self.config.get('patience', 3):
#                     self.logger.info("🛑 Early stopping فعال شد!")
#                     break
#
#         self.logger.info("\n" + "=" * 50)
#         self.logger.info("آموزش پایان یافت!")
#         self.logger.info(f"بهترین F1: {self.best_f1:.4f}")
#         self.logger.info("=" * 50)
#
#         # بارگذاری بهترین مدل
#         self._load_best_model()
#
#     def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
#         """ذخیره checkpoint بهترین مدل"""
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': self.model.classifier.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'scheduler_state_dict': self.scheduler.state_dict(),
#             'best_f1': self.best_f1,
#             'metrics': metrics,
#             'config': self.config
#         }
#         torch.save(checkpoint, 'best_model.pth')
#
#     def _load_best_model(self):
#         """بارگذاری بهترین مدل ذخیره‌شده"""
#         checkpoint = torch.load('best_model.pth')
#         self.model.classifier.load_state_dict(checkpoint['model_state_dict'])
#         self.logger.info("بهترین مدل بارگذاری شد.")
#
#     def train_anomaly_detector(self, normal_dataset):
#         """
#         آموزش شاخه آنومالی‌دیتکشن (بعد از آموزش classifier)
#         """
#         self.logger.info("\n" + "=" * 50)
#         self.logger.info("شروع آموزش Anomaly Detector")
#         self.logger.info("=" * 50)
#
#         # استخراج ویژگی‌های نرمال
#         self.model.classifier.eval()
#         normal_features = []
#
#         with torch.no_grad():
#             for batch in DataLoader(normal_dataset, batch_size=32):
#                 input_ids = batch['input_ids'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 word2vec = batch['word2vec_embeds'].to(self.device)
#                 fasttext = batch['fasttext_embeds'].to(self.device)
#
#                 outputs = self.model.classifier(
#                     input_ids, attention_mask, word2vec, fasttext
#                 )
#                 normal_features.append(outputs['features'].cpu().numpy())
#
#         normal_features = np.concatenate(normal_features, axis=0)
#
#         # آموزش آنومالی‌دیتکشن
#         self.model.train_anomaly_detector(normal_features)
#
#         self.logger.info("✅ Anomaly Detector آموزش دید!")
#         self.logger.info(f"Threshold: {self.model.anomaly_detector.reconstruction_threshold:.4f}")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from typing import Dict, Any
import logging


class VulnDetectionTrainer:
    def __init__(self, model, config: Dict[str, Any],
                 train_dataset, val_dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.config = config

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['classification']['batch_size'],  # ✅ تغییر
            shuffle=True,
            num_workers=2
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['classification']['batch_size'],  # ✅ تغییر
            shuffle=False,
            num_workers=2
        )

        # Optimizer برای CodeBERT (نه Sec-BERT)
        # ✅ تغییر: classifier.named_parameters() → codebert
        # codebert_params = [p for n, p in model.classifier.named_parameters() if 'codebert' in n]
        # other_params = [p for n, p in model.classifier.named_parameters() if 'codebert' not in n]
        #
        # self.optimizer = AdamW([
        #     {'params': codebert_params, 'lr': config['classification']['learning_rate']},
        #     {'params': other_params, 'lr': config['classification']['learning_rate'] * 10}
        # ], weight_decay=config['classification']['weight_decay'])
        # codebert_params = self.model.classifier.codebert.parameters()

        codebert_params = list(self.model.classifier.codebert.parameters())

        print("Trainable CodeBERT params:", sum(p.numel() for p in codebert_params if p.requires_grad))

        other_params = (
                list(self.model.classifier.embedding_fusion.parameters()) +
                list(self.model.classifier.feature_extractor.parameters()) +
                list(self.model.classifier.classifier.parameters())
        )

        self.optimizer = AdamW([
            {'params': codebert_params, 'lr': float(config['classification']['learning_rate'])},
            {'params': other_params, 'lr': float(config['classification']['learning_rate']) * 10}
        ], weight_decay=float(config['classification']['weight_decay']))
        #print("CodeBERT params:", sum(p.numel() for p in self.model.codebert.parameters() if p.requires_grad))
        #print("Optimizer params:", sum(p.numel() for g in self.optimizer.param_groups for p in g['params']))

        # Scheduler
        total_steps = len(self.train_loader) * config['classification']['epochs']  # ✅ تغییر
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['classification']['warmup_steps'],  # ✅ تغییر
            num_training_steps=total_steps
        )

        # Early stopping
        self.best_f1 = 0
        self.patience_counter = 0

        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def train_epoch(self, epoch: int) -> float:
        """آموزش یک epoch کامل (فقط classifier)"""
        self.model.classifier.train()
        total_loss = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # انتقال به device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            word2vec = batch['word2vec_embeds'].to(self.device)
            fasttext = batch['fasttext_embeds'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward
            outputs = self.model.classifier(
                input_ids, attention_mask, word2vec, fasttext, labels
            )

            loss = outputs['loss']

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping - فقط classifier
            torch.nn.utils.clip_grad_norm_(
                self.model.classifier.parameters(),
                self.config['classification']['max_grad_norm']  # ✅ تغییر
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                self.logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def validate(self) -> Dict[str, float]:
        """ارزیابی روی داده‌های اعتبارسنجی"""
        self.model.classifier.eval()
        all_preds = []
        all_labels = []
        all_losses = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                word2vec = batch['word2vec_embeds'].to(self.device)
                fasttext = batch['fasttext_embeds'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model.classifier(
                    input_ids, attention_mask, word2vec, fasttext, labels
                )

                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_losses.append(outputs['loss'].item())

        # محاسبه معیارها
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        accuracy = accuracy_score(all_labels, all_preds)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': np.mean(all_losses)
        }

    def train(self):
        """آموزش کامل مدل"""
        self.logger.info("=" * 50)
        self.logger.info("شروع آموزش مدل تشخیص آسیب‌پذیری")
        self.logger.info("=" * 50)

        for epoch in range(self.config['classification']['epochs']):
            self.logger.info(f"\n{'=' * 20} Epoch {epoch + 1}/{self.config['classification']['epochs']} {'=' * 20}")

            # آموزش
            train_loss = self.train_epoch(epoch)
            self.logger.info(f"Train Loss: {train_loss:.4f}")

            # اعتبارسنجی
            metrics = self.validate()
            self.logger.info(f"Val Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"Val F1: {metrics['f1']:.4f}")
            self.logger.info(f"Val Loss: {metrics['loss']:.4f}")

            # Early stopping
            if metrics['f1'] > self.best_f1:
                self.best_f1 = metrics['f1']
                self.patience_counter = 0
                self._save_checkpoint(epoch, metrics)
                self.logger.info("✅ مدل بهتر ذخیره شد!")
            else:
                self.patience_counter += 1
                self.logger.info(f"📉 Early stopping counter: {self.patience_counter}/{self.config.get('patience', 3)}")

                if self.patience_counter >= self.config.get('patience', 3):
                    self.logger.info("🛑 Early stopping فعال شد!")
                    break

        self.logger.info("\n" + "=" * 50)
        self.logger.info("آموزش پایان یافت!")
        self.logger.info(f"بهترین F1: {self.best_f1:.4f}")
        self.logger.info("=" * 50)

        # بارگذاری بهترین مدل
        self._load_best_model()

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """ذخیره checkpoint بهترین مدل"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': self.best_f1,
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, 'best_model.pth')

    def _load_best_model(self):
        """بارگذاری بهترین مدل ذخیره‌شده"""
        checkpoint = torch.load('best_model.pth')
        self.model.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info("بهترین مدل بارگذاری شد.")

    def train_anomaly_detector(self, normal_dataset):
        """
        آموزش شاخه آنومالی‌دیتکشن (بعد از آموزش classifier)
        """
        self.logger.info("\n" + "=" * 50)
        self.logger.info("شروع آموزش Anomaly Detector")
        self.logger.info("=" * 50)

        # استخراج ویژگی‌های نرمال
        self.model.classifier.eval()
        normal_features = []

        with torch.no_grad():
            for batch in DataLoader(normal_dataset, batch_size=32):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                word2vec = batch['word2vec_embeds'].to(self.device)
                fasttext = batch['fasttext_embeds'].to(self.device)

                outputs = self.model.classifier(
                    input_ids, attention_mask, word2vec, fasttext
                )
                normal_features.append(outputs['features'].cpu().numpy())

        normal_features = np.concatenate(normal_features, axis=0)

        # آموزش آنومالی‌دیتکشن
        self.model.train_anomaly_detector(normal_features)

        self.logger.info("✅ Anomaly Detector آموزش دید!")
        self.logger.info(f"Threshold: {self.model.anomaly_detector.reconstruction_threshold:.4f}")