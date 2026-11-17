import os
from pathlib import Path
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

CACHE_DIR = Path(__file__).resolve().parent.parent / "huggingface_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from sentence_transformers import SentenceTransformer

def extract_bow(
    texts,
    max_features: int = 12000,
    min_df: int = 3,
    ngram_range: tuple = (1, 2),
    stop_words: str | list | None = "english",
    **kwargs
):
    """
    Trích xuất vector đặc trưng Bag-of-Words (BoW):
    - max_features=12000, min_df=3 để loại bỏ từ hiếm/nhiễu
    - ngram_range=(1, 2) thu thập cả unigram và bigram
    - stop_words='english' loại bỏ từ dừng tiếng Anh phổ biến
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
        stop_words=stop_words,
        **kwargs
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def extract_tfidf(
    texts,
    max_features: int = 12000,
    min_df: int = 3,
    ngram_range: tuple = (1, 2),
    stop_words: str | list | None = "english",
    **kwargs
):
    """
    Trích xuất vector đặc trưng TF-IDF:
    - max_features=12000, min_df=3
    - ngram_range=(1, 2)
    - stop_words='english'
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
        stop_words=stop_words,
        **kwargs
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def extract_embeddings(
    texts,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 128,
    device: str | None = None
):
    """
    Trích xuất vector đặc trưng Embeddings ngữ nghĩa:
    - Tự động nhận diện GPU (cuda) nếu khả dụng để tối ưu tốc độ
    - batch_size=128 giúp tận dụng tối đa băng thông bộ nhớ
    - Sử dụng cache local đã tải sẵn trong huggingface_cache
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        model_name,
        cache_folder=str(CACHE_DIR / "hub"),
        device=device
    )
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        device=device,
        show_progress_bar=False
    )
    return embeddings, model

if __name__ == "__main__":
    sample_texts = [
        "deep learning and machine learning in 3D quantum computing",
        "quantum physics and particle accelerator collider at 14 TeV",
        "astronomy observations of distant galaxies in 2024"
    ]

    # Kiểm thử BoW (với min_df=1 cho sample nhỏ)
    X_bow, _ = extract_bow(sample_texts, min_df=1)
    print(f"BoW shape: {X_bow.shape}")

    # Kiểm thử TF-IDF (với min_df=1 cho sample nhỏ)
    X_tfidf, _ = extract_tfidf(sample_texts, min_df=1)
    print(f"TF-IDF shape: {X_tfidf.shape}")

    # Kiểm thử Embeddings
    device_used = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kiểm thử Embeddings trên thiết bị: {device_used}")
    X_emb, _ = extract_embeddings(sample_texts, batch_size=128)
    print(f"Embeddings shape: {X_emb.shape}")
    assert X_emb.shape == (len(sample_texts), 384)
    print("-> Tất cả kiểm thử vector hóa đã thành công!")
