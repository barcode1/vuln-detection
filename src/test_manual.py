import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
import yaml
import os
from src.models.ensemble import EnsembleVulnDetector
from src.data_pipeline.preprocessor import SecurityPreprocessor
from src.data_pipeline.tokenizer import MultiEmbeddingTokenizer

# دیتاست تست (همان ساختار VulnDataset)
class TestVulnDataset(Dataset):
    def __init__(self, sec_bert_encodings, word2vec, fasttext, labels=None):
        self.sec_bert = sec_bert_encodings
        self.word2vec = word2vec
        self.fasttext = fasttext
        self.labels = labels  # می‌تونه None باشه اگر unlabeled باشه

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.sec_bert.items()}
        item['word2vec_embeds'] = torch.tensor(self.word2vec[idx])
        item['fasttext_embeds'] = torch.tensor(self.fasttext[idx])
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.word2vec)


def load_model(config_path: str, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """بارگذاری مدل آموزش‌دیده"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model = EnsembleVulnDetector(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # مهم: حالت ارزیابی برای فعال شدن آنومالی
    print(f"✅ loaded model in: {model_path}")
    print(f"   divice: {device}")
    return model, config


def test_model(
    model,
    config,
    csv_path: str,
    text_column: str = 'Sentence',
    label_column: str = 'label',  # یا ستون‌های باینری مثل SQLInjection, XSS, ...
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """تست کامل مدل روی یک دیتاست CSV"""
    print("\n" + "="*60)
    print("🚀 start test model...")
    print("="*60)

    # خواندن دیتاست
    df = pd.read_csv(csv_path)
    print(f"count of sample: {len(df)}")

    # تبدیل برچسب‌ها اگر باینری باشه
    if label_column not in df.columns:
        print("ستون label وجود ندارد → تبدیل از ستون‌های باینری...")
        def get_label(row):
            if row.get('SQLInjection', 0) == 1: return 1
            elif row.get('XSS', 0) == 1: return 2
            elif row.get('CommandInjection', 0) == 1: return 3
            elif row.get('Normal', 0) == 1: return 0
            else: return -1
        df['label'] = df.apply(get_label, axis=1)
        df = df[df['label'] != -1].reset_index(drop=True)
        label_column = 'label'

    texts = df[text_column].tolist()
    labels = df[label_column].values if label_column in df.columns else None

    # پیش‌پردازش
    preprocessor = SecurityPreprocessor(config.get('preprocessing', {}))
    processed_texts = preprocessor.fit_transform(texts)

    # توکنایزر
    tokenizer = MultiEmbeddingTokenizer(config)
    tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'embeddings')
    tokenizer.load_word_embeddings(tokenizer_path)

    # جاسازی‌ها
    embeddings = tokenizer.encode(processed_texts)

    # دیتاست و دیتالودر
    dataset = TestVulnDataset(
        embeddings['sec_bert'],
        embeddings['word2vec'].numpy(),
        embeddings['fasttext'].numpy(),
        labels
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # پیش‌بینی‌ها
    all_preds = []
    all_labels = []
    all_anomaly_scores = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            word2vec = batch['word2vec_embeds'].to(device)
            fasttext = batch['fasttext_embeds'].to(device)

            outputs = model(input_ids, attention_mask, word2vec, fasttext)

            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            anomaly_scores = outputs['anomaly_scores'].cpu().numpy().flatten()

            all_preds.extend(preds)
            all_anomaly_scores.extend(anomaly_scores)
            if 'labels' in batch:
                all_labels.extend(batch['labels'].cpu().numpy())

    # گزارش نتایج
    print("\n" + "="*60)
    print("📊 نتایج تست")
    print("="*60)

    if labels is not None:
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
        class_names = ['Normal', 'SQLi', 'XSS', 'CMDi']

        print(f"دقت کلی (Accuracy): {accuracy:.4f}")
        print("\nreport for any class::")
        for i, name in enumerate(class_names):
            print(f"   {name:8} → Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f} | F1: {f1[i]:.4f}")

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

        print("\n(Confusion Matrix):")
        cm = confusion_matrix(all_labels, all_preds)
        print("       Pred →  Normal  SQLi   XSS   CMDi")
        for i, row in enumerate(cm):
            print(f"True {class_names[i]:6} → {row}")

        # نمونه‌های اشتباه با تشخیص آنومالی
        print("\n🔍 ۱۰ نمونه اشتباه (با امتیاز آنومالی):")
        errors = np.where(np.array(all_preds) != np.array(all_labels))[0]
        for idx in errors[:10]:
            text = texts[idx]
            true = class_names[all_labels[idx]]
            pred = class_names[all_preds[idx]]
            anomaly = all_anomaly_scores[idx]
            print(f"   متن: {text[:80]}{'...' if len(text)>80 else ''}")
            print(f"   true: {true} | predection: {pred} | anomaly: {anomaly:.2f}")
            print("   ---")
    else:
        print("دیتاست بدون برچسب → فقط پیش‌بینی انجام شد.")
        for i in range(min(10, len(texts))):
            text = texts[i]
            pred = ['Normal', 'SQLi', 'XSS', 'CMDi'][all_preds[i]]
            anomaly = all_anomaly_scores[i]
            print(f"   متن: {text[:80]}...")
            print(f"   predection: {pred} | score anomaly: {anomaly:.2f}")
            print("   ---")

    print(f"\n ave score anomaly in total dataset: {np.mean(all_anomaly_scores):.4f}")
    print("="*60)
    print("✅ test finished...!")


# استفاده ساده
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'hyperparams.yaml')
    MODEL_PATH = os.path.join(BASE_DIR, 'final_model.pth')
    TEST_CSV_PATH = os.path.join(BASE_DIR, 'data', 'datasets', 'SQLInjection_XSS_CommandInjection_MixDataset.1.0.0.csv')  # مسیر دیتاست تست

    model, config = load_model(CONFIG_PATH, MODEL_PATH)

    test_model(
        model=model,
        config=config,
        csv_path=TEST_CSV_PATH,
        text_column='Sentence',
        label_column='label',
        batch_size=32
    )