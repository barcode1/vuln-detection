import yaml
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.models.ensemble import EnsembleVulnDetector
from src.training.trainer import VulnDetectionTrainer
from src.data_pipeline.preprocessor import SecurityPreprocessor
from src.data_pipeline.tokenizer import MultiEmbeddingTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class VulnDataset(torch.utils.data.Dataset):
    def __init__(self, sec_bert_encodings, word2vec, fasttext, labels):
        self.sec_bert = sec_bert_encodings
        self.word2vec = word2vec
        self.fasttext = fasttext
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.sec_bert.items()}
        item['word2vec_embeds'] = torch.tensor(self.word2vec[idx])
        item['fasttext_embeds'] = torch.tensor(self.fasttext[idx])
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def create_single_label(df):
    """
    تبدیل 4 ستون باینری (SQLInjection, XSS, CommandInjection, Normal)
    به یک ستون واحد label (0, 1, 2, 3)
    """

    # تعریف تابع برای هر ردیف
    def get_label(row):
        if row['SQLInjection'] == 1.0:
            return 1  # SQLi
        elif row['XSS'] == 1.0:
            return 2  # XSS
        elif row['CommandInjection'] == 1.0:
            return 3  # CMDi
        elif row['Normal'] == 1.0:
            return 0  # Normal
        else:
            return -1  # برچسب نامعتبر

    # اعمال تابع به تمام ردیف‌ها
    df['label'] = df.apply(get_label, axis=1)

    # حذف ردیف‌های دارای برچسب نامعتبر (اگر وجود داشت)
    df = df[df['label'] != -1].reset_index(drop=True)

    # تبدیل به integer
    df['label'] = df['label'].astype(int)

    return df


def main():
    # تنظیم مسیرها
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'hyperparams.yaml')
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'datasets', 'SQLInjection_XSS_CommandInjection_MixDataset.1.0.0.csv')

    # بارگذاری پیکربندی
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ✅ بارگذاری داده‌ها
    print("📂 Loading data...")
    df = pd.read_csv(DATA_PATH)

    # ✅ دیباگ: نمایش ساختار دیتا
    print("=" * 60)
    print("DEBUG: Dataset Structure")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("First 3 rows:")
    print(df.head(3))
    print("=" * 60)

    # ✅ تبدیل 4 ستون باینری به یک ستون واحد label
    print("🔧 Converting binary columns to single label...")
    df = create_single_label(df)

    # ✅ دیباگ: بررسی توزیع کلاس‌ها
    print("\nClass distribution:")
    print(df['label'].value_counts())
    print("=" * 60)

    # ✅ پیکربندی نام ستون‌ها
    TEXT_COLUMN = 'Sentence'  # اینجا دقیقاً نام ستون شماست
    LABEL_COLUMN = 'label'  # اینی که ساختیم

    # ✅ بررسی وجود ستون‌ها
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"ستون متن '{TEXT_COLUMN}' پیدا نشد!")
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"ستون برچسب '{LABEL_COLUMN}' پیدا نشد!")

    print(f"✅ Using text column: '{TEXT_COLUMN}'")
    print(f"✅ Using label column: '{LABEL_COLUMN}'")

    # ✅ preprocessing
    preprocessor = SecurityPreprocessor(config.get('preprocessing', {}))
    processed_texts = preprocessor.fit_transform(df[TEXT_COLUMN].tolist())

    # ✅ train or loading Word2Vec/FastText
    tokenizer = MultiEmbeddingTokenizer(config)
    tokenizer_path = os.path.join(BASE_DIR, 'data', 'embeddings')

    # if os.path.exists(os.path.join(tokenizer_path, 'word2vec_security.model')):
    #     tokenizer.load_word_embeddings(tokenizer_path)
    # else:
    #     os.makedirs(tokenizer_path, exist_ok=True)
    #     tokenizer.train_word_embeddings(processed_texts)
    #     tokenizer.save_word_embeddings(tokenizer_path)
    # در setup.py
    if os.path.exists(os.path.join(tokenizer_path, 'word2vec_security.model')):
        tokenizer.load_word_embeddings(tokenizer_path)
    else:
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.train_word_embeddings(processed_texts)
        tokenizer.save_word_embeddings(tokenizer_path)

    # ✅ generate embedding
    embeddings = tokenizer.encode(processed_texts)

    # ✅ splite data
    train_idx, val_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=42, stratify=df[LABEL_COLUMN]
    )

    train_dataset = VulnDataset(
        {k: v[train_idx] for k, v in embeddings['sec_bert'].items()},
        embeddings['word2vec'][train_idx].numpy(),
        embeddings['fasttext'][train_idx].numpy(),
        df[LABEL_COLUMN].iloc[train_idx].values
    )

    val_dataset = VulnDataset(
        {k: v[val_idx] for k, v in embeddings['sec_bert'].items()},
        embeddings['word2vec'][val_idx].numpy(),
        embeddings['fasttext'][val_idx].numpy(),
        df[LABEL_COLUMN].iloc[val_idx].values
    )

    # ✅ create model and train
    model = EnsembleVulnDetector(config)
    trainer = VulnDetectionTrainer(model, config, train_dataset, val_dataset)
    trainer.train()
    # ==================== train Anomaly Detector ====================
    print("\n" + "=" * 60)
    print("🚀 start training anomaly_detection on sample Normal...")
    print("=" * 60)

    # just data Normal (label == 0)
    normal_df = df[df['label'] == 0].reset_index(drop=True)

    if len(normal_df) == 0:
        raise ValueError("هیچ نمونه Normal در دیتاست پیدا نشد!")

    print(f"count sample normal for anomaly: {len(normal_df)}")

    #preprocess for anomaly_detection
    normal_texts = preprocessor.fit_transform(normal_df[TEXT_COLUMN].tolist())

    # generate embedding
    normal_embeddings = tokenizer.encode(normal_texts)

    # create dataset just Normal
    normal_dataset = VulnDataset(
        {k: v for k, v in normal_embeddings['sec_bert'].items()},
        normal_embeddings['word2vec'].numpy(),
        normal_embeddings['fasttext'].numpy(),
        normal_df[LABEL_COLUMN].values
    )

    # train Anomaly Detector
    trainer.train_anomaly_detector(normal_dataset)

    print("✅ Anomaly Detector train !")
    print(f"   Reconstruction Threshold: {model.anomaly_detector.reconstruction_threshold:.4f}")
    # ✅ ذخیره مدل
    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'final_model.pth'))
    print("✅ finally model saved !")


if __name__ == '__main__':
    main()