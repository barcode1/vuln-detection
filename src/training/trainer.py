import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from typing import Dict, Any
import logging
import time
from datetime import timedelta
from tqdm import tqdm

class VulnDetectionTrainer:
    def __init__(self, model, config: Dict[str, Any],
                 train_dataset, val_dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.config = config

        # DataLoaders
        # self.train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=config['classification']['batch_size'],  # ✅ تغییر
        #     shuffle=True,
        #     num_workers=2
        # )
        # self.val_loader = DataLoader(
        #     val_dataset,
        #     batch_size=config['classification']['batch_size'],  # ✅ تغییر
        #     shuffle=False,
        #     num_workers=2
        # )
        # ✅ DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['classification']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=config['classification'].get('pin_memory', False),
            persistent_workers=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['classification']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=config['classification'].get('pin_memory', False),
            persistent_workers=True
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

    def debug_dataloader_speed(self):
        """تست سرعت DataLoader"""
        print("\n🔍 تست سرعت DataLoader...")

        start = time.time()
        for i, batch in enumerate(self.train_loader):
            if i == 10:  # فقط ۱۰ بچ تست کن
                break
            load_time = time.time() - start
            print(f"Batch {i}: {load_time:.2f}s")
            start = time.time()

    # def train_epoch(self, epoch: int) -> float:
    #     """آموزش یک epoch کامل (فقط classifier)"""
    #     self.model.classifier.train()
    #     total_loss = 0
    #
    #     # ✅ Progress bar برای آموزش
    #     pbar = tqdm(self.train_loader,
    #                 desc=f"🚀 Epoch {epoch + 1} | Training",
    #                 ncols=100)
    #     self.debug_dataloader_speed()
    #     for batch_idx, batch in enumerate(self.train_loader):
    #         # انتقال به device
    #         input_ids = batch['input_ids'].to(self.device)
    #         attention_mask = batch['attention_mask'].to(self.device)
    #         word2vec = batch['word2vec_embeds'].to(self.device)
    #         fasttext = batch['fasttext_embeds'].to(self.device)
    #         labels = batch['labels'].to(self.device)
    #
    #         # Forward
    #         outputs = self.model.classifier(
    #             input_ids, attention_mask, word2vec, fasttext, labels
    #         )
    #
    #         loss = outputs['loss']
    #
    #         # Backward
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #
    #         # Gradient clipping - فقط classifier
    #         torch.nn.utils.clip_grad_norm_(
    #             self.model.classifier.parameters(),
    #             self.config['classification']['max_grad_norm']  # ✅ تغییر
    #         )
    #
    #         self.optimizer.step()
    #         self.scheduler.step()
    #
    #         total_loss += loss.item()
    #
    #
    #         if batch_idx % 100 == 0:
    #             self.logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
    #         pbar.set_postfix({
    #             'Loss': f'{loss.item():.4f}',
    #             'Avg': f'{total_loss / (batch_idx + 1):.4f}'
    #         })
    #     return total_loss / len(self.train_loader)
    def train_epoch(self, epoch: int) -> float:
        """آموزش یک epoch کامل (فقط classifier)"""
        self.model.classifier.train()
        total_loss = 0

        # ✅ حذف یا کامنت کن debug_dataloader_speed
        # self.debug_dataloader_speed()  # ❌ این خط رو پاک کن یا کامنت کن

        # ✅ Progress bar برای آموزش
        pbar = tqdm(self.train_loader,
                    desc=f"🚀 Epoch {epoch + 1} | Training",
                    ncols=100,
                    total=len(self.train_loader))  # ✅ اضافه کن total

        # ✅ **مهم**: از pbar استفاده کن نه self.train_loader
        for batch_idx, batch in enumerate(pbar):
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

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.classifier.parameters(),
                self.config['classification']['max_grad_norm']
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            # ✅ آپدیت progress bar
            if batch_idx % 10 == 0:  # هر ۱۰ بچ آپدیت کن (کمترین overhead)
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{total_loss / (batch_idx + 1):.4f}'
                })

            # ✅ اثری نیست: هر ۱۰۰ بچ log کن (اختیاری)
            if batch_idx % 100 == 0:
                self.logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")

        return total_loss / len(self.train_loader)

    # def validate(self) -> Dict[str, float]:
    #     """ارزیابی روی داده‌های اعتبارسنجی"""
    #     self.model.classifier.eval()
    #     all_preds = []
    #     all_labels = []
    #     all_losses = []
    #
    #     with torch.no_grad():
    #         for batch in self.val_loader:
    #             input_ids = batch['input_ids'].to(self.device)
    #             attention_mask = batch['attention_mask'].to(self.device)
    #             word2vec = batch['word2vec_embeds'].to(self.device)
    #             fasttext = batch['fasttext_embeds'].to(self.device)
    #             labels = batch['labels'].to(self.device)
    #
    #             outputs = self.model.classifier(
    #                 input_ids, attention_mask, word2vec, fasttext, labels
    #             )
    #
    #             logits = outputs['logits']
    #             preds = torch.argmax(logits, dim=1)
    #
    #             all_preds.extend(preds.cpu().numpy())
    #             all_labels.extend(labels.cpu().numpy())
    #             all_losses.append(outputs['loss'].item())
    #
    #     # محاسبه معیارها
    #     precision, recall, f1, _ = precision_recall_fscore_support(
    #         all_labels, all_preds, average='weighted', zero_division=0
    #     )
    #     accuracy = accuracy_score(all_labels, all_preds)
    #     torch.cuda.empty_cache()
    #
    #     return {
    #         'accuracy': accuracy,
    #         'precision': precision,
    #         'recall': recall,
    #         'f1': f1,
    #         'loss': np.mean(all_losses)
    #     }
    def validate(self) -> Dict[str, float]:
        """ارزیابی روی داده‌های اعتبارسنجی"""
        self.model.classifier.eval()
        all_preds = []
        all_labels = []
        all_losses = []

        # ✅ Progress bar برای اعتبارسنجی
        pbar = tqdm(self.val_loader,
                    desc="  🔍 Validation",
                    ncols=100,
                    total=len(self.val_loader))

        with torch.no_grad():
            for batch in pbar:  # ✅ از pbar استفاده کن
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

                # ✅ آپدیت progress bar
                pbar.set_postfix({'Loss': f'{outputs["loss"].item():.4f}'})

        # محاسبه معیارها
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        accuracy = accuracy_score(all_labels, all_preds)
        torch.cuda.empty_cache()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': np.mean(all_losses)
        }

    def train(self):
        """Full model training with timing and metrics"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Vulnerability Detection Model Training")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("=" * 60)

        start_time = time.time()

        try:
            for epoch in range(self.config['classification']['epochs']):
                epoch_start = time.time()

                # Training
                self.logger.info(
                    f"\nEpoch {epoch + 1}/{self.config['classification']['epochs']} - Starting Training...")
                train_loss = self.train_epoch(epoch)
                train_time = time.time() - epoch_start

                # Validation
                self.logger.info(f"Starting Validation...")
                metrics = self.validate()
                epoch_time = time.time() - epoch_start

                # Display results
                print("\n" + "=" * 70)
                print(f"Epoch {epoch + 1}/{self.config['classification']['epochs']} Results")
                print("=" * 70)
                print(f"Total Time:     {str(timedelta(seconds=int(epoch_time)))}")
                print(f"  └─ Training:  {str(timedelta(seconds=int(train_time)))}")
                print(f"  └─ Val:       {str(timedelta(seconds=int(epoch_time - train_time)))}")
                print("─" * 70)
                print(f"Train Loss:     {train_loss:.4f}")
                print(f"Val Accuracy:   {metrics['accuracy']:.4f}")
                print(f"Val F1-Score:   {metrics['f1']:.4f}")
                print(f"Val Loss:       {metrics['loss']:.4f}")
                print("=" * 70 + "\n")

                # Early stopping
                if metrics['f1'] > self.best_f1:
                    self.best_f1 = metrics['f1']
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, metrics)
                    self.logger.info("Best model saved!")
                else:
                    self.patience_counter += 1
                    self.logger.info(
                        f"Early stopping counter: {self.patience_counter}/{self.config.get('patience', 3)}")

                    if self.patience_counter >= self.config.get('patience', 3):
                        self.logger.info("Early stopping triggered!")
                        break

        except KeyboardInterrupt:
            self.logger.info("\nTraining interrupted by user!")
        except Exception as e:
            self.logger.error(f"Error occurred: {str(e)}")
            raise
        finally:
            # Load best model (fix for PyTorch 2.6+)
            try:
                # Allow numpy objects in checkpoint (safe for own checkpoints)
                import numpy
                torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
                checkpoint = torch.load('best_model.pth', weights_only=True)
            except (AttributeError, TypeError, pickle.UnpicklingError):
                # Fallback for compatibility
                checkpoint = torch.load('best_model.pth', weights_only=False)

            self.model.classifier.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Best model loaded successfully.")

            # Final summary
            total_time = time.time() - start_time
            self.logger.info("\n" + "=" * 60)
            self.logger.info("Training Completed!")
            self.logger.info(f"Total Time: {str(timedelta(seconds=int(total_time)))}")
            self.logger.info(f"Best F1: {self.best_f1:.4f}")
            self.logger.info("=" * 60)
    #
    # def train(self):
    #     """Full model training with timing and metrics"""
    #     self.logger.info("=" * 60)
    #     self.logger.info("Starting Vulnerability Detection Model Training")
    #     self.logger.info(f"Device: {self.device}")
    #     self.logger.info("=" * 60)
    #
    #     start_time = time.time()
    #
    #     try:
    #         for epoch in range(self.config['classification']['epochs']):
    #             epoch_start = time.time()
    #
    #             # Training
    #             self.logger.info(
    #                 f"\nEpoch {epoch + 1}/{self.config['classification']['epochs']} - Starting Training...")
    #             train_loss = self.train_epoch(epoch)
    #             train_time = time.time() - epoch_start
    #
    #             # Validation
    #             self.logger.info(f"Starting Validation...")
    #             metrics = self.validate()
    #             epoch_time = time.time() - epoch_start
    #
    #             # Display results
    #             print("\n" + "=" * 70)
    #             print(f"Epoch {epoch + 1}/{self.config['classification']['epochs']} Results")
    #             print("=" * 70)
    #             print(f"Total Time:     {str(timedelta(seconds=int(epoch_time)))}")
    #             print(f"  └─ Training:  {str(timedelta(seconds=int(train_time)))}")
    #             print(f"  └─ Val:       {str(timedelta(seconds=int(epoch_time - train_time)))}")
    #             print("─" * 70)
    #             print(f"Train Loss:     {train_loss:.4f}")
    #             print(f"Val Accuracy:   {metrics['accuracy']:.4f}")
    #             print(f"Val F1-Score:   {metrics['f1']:.4f}")
    #             print(f"Val Loss:       {metrics['loss']:.4f}")
    #             print("=" * 70 + "\n")
    #
    #             # Early stopping
    #             if metrics['f1'] > self.best_f1:
    #                 self.best_f1 = metrics['f1']
    #                 self.patience_counter = 0
    #                 self._save_checkpoint(epoch, metrics)
    #                 self.logger.info("Best model saved!")
    #             else:
    #                 self.patience_counter += 1
    #                 self.logger.info(
    #                     f"Early stopping counter: {self.patience_counter}/{self.config.get('patience', 3)}")
    #
    #                 if self.patience_counter >= self.config.get('patience', 3):
    #                     self.logger.info("Early stopping triggered!")
    #                     break
    #
    #     except KeyboardInterrupt:
    #         self.logger.info("\nTraining interrupted by user!")
    #     except Exception as e:
    #         self.logger.error(f"Error occurred: {str(e)}")
    #         raise
    #     finally:
    #         # Final summary
    #         total_time = time.time() - start_time
    #         self.logger.info("\n" + "=" * 60)
    #         self.logger.info("Training Completed!")
    #         self.logger.info(f"Total Time: {str(timedelta(seconds=int(total_time)))}")
    #         self.logger.info(f"Best F1: {self.best_f1:.4f}")
    #         self.logger.info("=" * 60)
    #
    #         # Load best model
    #         self._load_best_model()

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
        self.logger.info("loaded the best model.")

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
            for batch in DataLoader(
            normal_dataset,
            batch_size=256,
            num_workers=8,
            pin_memory=True):
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

        self.logger.info("✅ Anomaly Detector trained!")
        self.logger.info(f"Threshold: {self.model.anomaly_detector.reconstruction_threshold:.4f}")