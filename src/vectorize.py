import os
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

CACHE_DIR = Path(__file__).resolve().parent.parent / "huggingface_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from sentence_transformers import SentenceTransformer

def extract_bow(texts, max_features: int = 5000, **kwargs):
    """Trích xuất vector đặc trưng Bag-of-Words (BoW) bằng CountVectorizer."""
    vectorizer = CountVectorizer(max_features=max_features, **kwargs)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def extract_tfidf(texts, max_features: int = 5000, **kwargs):
    """Trích xuất vector đặc trưng TF-IDF bằng TfidfVectorizer."""
    vectorizer = TfidfVectorizer(max_features=max_features, **kwargs)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def extract_embeddings(
    texts,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 32
):
    """Trích xuất vector đặc trưng Embeddings từ mô hình đa ngôn ngữ được cache local."""
    model = SentenceTransformer(model_name, cache_folder=str(CACHE_DIR / "hub"))
    embeddings = model.encode(list(texts), batch_size=batch_size, show_progress_bar=False)
    return embeddings, model

if __name__ == "__main__":
    sample_texts = [
        "học_máy là một nhánh của trí_tuệ nhân_tạo",
        "vector_hóa văn_bản với tfidf và embeddings",
        "phân loại văn bản khoa học"
    ]

    # Kiểm thử BoW
    X_bow, bow_vec = extract_bow(sample_texts)
    assert X_bow.shape[0] == len(sample_texts)
    print(f"BoW shape: {X_bow.shape}")

    # Kiểm thử TF-IDF
    X_tfidf, tfidf_vec = extract_tfidf(sample_texts)
    assert X_tfidf.shape[0] == len(sample_texts)
    print(f"TF-IDF shape: {X_tfidf.shape}")

    # Kiểm thử Embeddings
    X_emb, emb_model = extract_embeddings(sample_texts)
    assert X_emb.shape == (len(sample_texts), 384)
    print(f"Embeddings shape: {X_emb.shape}")
    print("Tất cả kiểm thử vector hóa đã pass thành công!")
