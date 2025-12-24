# src/main.py کامل و صحیح
import os
import yaml
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# واردات از ماژول‌های خودتان
from src.models.ensemble import EnsembleVulnDetector
from src.training.trainer import VulnDetectionTrainer
from src.data_pipeline.preprocessor import SecurityPreprocessor
from src.data_pipeline.tokenizer import MultiEmbeddingTokenizer


# کلاس Dataset
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
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def main():
    # تنظیم مسیرها
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'hyperparams.yaml')
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'datasets', 'SQLInjection_XSS_CommandInjection_MixDataset.1.0.0.csv')  # جایگزین کنید

    # بارگذاری پیکربندی
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # بارگذاری داده‌ها (مثال)
    df = pd.read_csv(DATA_PATH)

    # ✅ پیش‌پردازش
    preprocessor = SecurityPreprocessor(config.get('preprocessing', {}))
    processed_texts = preprocessor.fit_transform(df['Sentence'].tolist())

    # ✅ آموزش یا بارگذاری Word2Vec/FastText
    tokenizer = MultiEmbeddingTokenizer(config)
    tokenizer_path = os.path.join(BASE_DIR, 'data', 'embeddings')

    if os.path.exists(os.path.join(tokenizer_path, 'word2vec_security.model')):
        tokenizer.load_word_embeddings(tokenizer_path)
    else:
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.train_word_embeddings(processed_texts)
        tokenizer.save_word_embeddings(tokenizer_path)

    # ✅ تولید جاسازی‌ها (تک‌تک)
    embeddings = tokenizer.encode(processed_texts)
    # embeddings = {
    #     'sec_bert': {'input_ids': ..., 'attention_mask': ...},
    #     'word2vec': torch.tensor([...]),
    #     'fasttext': torch.tensor([...])
    # }

    # ✅ تقسیم داده‌ها
    train_idx, val_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=42, stratify=df['label']
    )

    train_dataset = VulnDataset(
        {k: v[train_idx] for k, v in embeddings['sec_bert'].items()},
        embeddings['word2vec'][train_idx].numpy(),
        embeddings['fasttext'][train_idx].numpy(),
        df['label'].iloc[train_idx].values
    )

    val_dataset = VulnDataset(
        {k: v[val_idx] for k, v in embeddings['sec_bert'].items()},
        embeddings['word2vec'][val_idx].numpy(),
        embeddings['fasttext'][val_idx].numpy(),
        df['label'].iloc[val_idx].values
    )

    # ✅ ساخت مدل
    model = EnsembleVulnDetector(config)

    # ✅ آموزش
    trainer = VulnDetectionTrainer(model, config, train_dataset, val_dataset)
    trainer.train()

    # ✅ ذخیره مدل
    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'final_model.pth'))
    print("✅ مدل نهایی ذخیره شد!")


if __name__ == '__main__':
    main()