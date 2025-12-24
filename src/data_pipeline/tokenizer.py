import torch
import numpy as np
import re
import os
from transformers import AutoTokenizer
from gensim.models import Word2Vec, FastText
from typing import List, Dict, Any


class MultiEmbeddingTokenizer:
    def __init__(self, config: Dict[str, Any]):
        # Sec-BERT Tokenizer
        self.sec_bert_tokenizer = AutoTokenizer.from_pretrained(
            config['sec_bert']['model_name']
        )

        # مدل‌های جاسازی
        self.word2vec = None
        self.fasttext = None

        # پارامترها
        self.max_length = config['data']['max_seq_length']
        self.embedding_dim = config['embedding']['embedding_dim']

        # مسیر ذخیره‌سازی (absolute path)
        self.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'embeddings'
        )

    def _simple_tokenize(self, text: str) -> List[str]:
        """تجزیه هوشمند برای Word2Vec/FastText"""
        # حفظ کلمات کلیدی مخرب و نمادها
        # مثلاً: "SELECT * FROM users" → ["SELECT", "*", "FROM", "users"]
        # مثلاً: "<script>alert(1)</script>" → ["<script>", "alert", "(", "1", ")</script"]
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return tokens if tokens else ['<PAD>']

    def train_word_embeddings(self, corpus: List[str]):
        """آموزش و ذخیره Word2Vec و FastText"""
        os.makedirs(self.save_dir, exist_ok=True)

        tokenized_corpus = [self._simple_tokenize(text) for text in corpus]

        # آموزش Word2Vec
        self.word2vec = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=self.embedding_dim,
            window=5,
            min_count=2,
            epochs=50,
            sg=1,
            workers=4
        )
        self.word2vec.save(os.path.join(self.save_dir, 'word2vec_security.model'))

        # آموزش FastText
        self.fasttext = FastText(
            sentences=tokenized_corpus,
            vector_size=self.embedding_dim,
            window=5,
            min_count=2,
            epochs=50,
            sg=1,
            workers=4
        )
        self.fasttext.save(os.path.join(self.save_dir, 'fasttext_security.model'))

        print(f"✅ مدل‌ها ذخیره شدند در: {self.save_dir}")

    def load_word_embeddings(self):
        """بارگذاری مدل‌های آموزش‌دیده"""
        w2v_path = os.path.join(self.save_dir, 'word2vec_security.model')
        ft_path = os.path.join(self.save_dir, 'fasttext_security.model')

        if os.path.exists(w2v_path) and os.path.exists(ft_path):
            self.word2vec = Word2Vec.load(w2v_path)
            self.fasttext = FastText.load(ft_path)
            print(f"✅ مدل‌ها بارگذاری شدند از: {self.save_dir}")
        else:
            raise FileNotFoundError(f"مدل‌ها در {self.save_dir} یافت نشدند!")

    def encode(self, texts: List[str]) -> Dict[str, Any]:
        """تولید 3 نوع جاسازی برای هر متن"""
        return {
            'sec_bert': self._encode_sec_bert(texts),
            'word2vec': self._encode_word2vec(texts),
            'fasttext': self._encode_fasttext(texts)
        }

    def _encode_sec_bert(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """جاسازی Sec-BERT با WordPiece"""
        return self.sec_bert_tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True
        )

    def _encode_word2vec(self, texts: List[str]) -> torch.Tensor:
        """جاسازی میانگین Word2Vec برای هر متن"""
        embeddings = []
        for text in texts:
            tokens = self._simple_tokenize(text)
            token_vecs = [self.word2vec.wv[token] for token in tokens if token in self.word2vec.wv]
            embeddings.append(np.mean(token_vecs, axis=0) if token_vecs else np.zeros(self.embedding_dim))
        return torch.FloatTensor(np.array(embeddings))

    def _encode_fasttext(self, texts: List[str]) -> torch.Tensor:
        """جاسازی میانگین FastText برای هر متن"""
        embeddings = []
        for text in texts:
            tokens = self._simple_tokenize(text)
            token_vecs = [self.fasttext.wv[token] for token in tokens]
            embeddings.append(np.mean(token_vecs, axis=0) if token_vecs else np.zeros(self.embedding_dim))
        return torch.FloatTensor(np.array(embeddings))