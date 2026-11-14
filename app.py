import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocess import clean_text, tokenize_text
from src.vectorize import extract_bow, extract_tfidf, extract_embeddings
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Cấu hình trang
st.set_page_config(page_title="Phân loại Chủ đề Bài báo ML", page_icon="📝", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

@st.cache_resource(show_spinner="Đang nạp dữ liệu và huấn luyện 7 mô hình...")
def load_and_train_models():
    """Huấn luyện sẵn các mô hình trên dữ liệu đã tiền xử lý để dự đoán tức thì."""
    if not DATA_PATH.exists():
        from src.load_data import load_data
        from src.preprocess import preprocess_dataset
        df = load_data(sample_size=1000)
        df = preprocess_dataset(df)
    else:
        df = pd.read_csv(DATA_PATH)

    top_labels = df["label"].value_counts().nlargest(12).index
    df_filtered = df[df["label"].isin(top_labels)].copy().reset_index(drop=True)
    texts = df_filtered["cleaned_text"].astype(str).tolist()
    y = np.array(df_filtered["label"].tolist())

    # 1. Trích xuất 3 bộ vector đặc trưng
    X_bow, bow_vec = extract_bow(texts)
    X_tfidf, tfidf_vec = extract_tfidf(texts)
    X_emb, emb_model = extract_embeddings(texts)

    # 2. Huấn luyện các mô hình tiêu biểu
    nb_bow = MultinomialNB().fit(X_bow, y)
    dt_tfidf = DecisionTreeClassifier(random_state=42).fit(X_tfidf, y)
    svm_tfidf = LinearSVC(dual="auto", random_state=42).fit(X_tfidf, y)
    knn_emb = KNeighborsClassifier(n_neighbors=5, metric="cosine").fit(X_emb, y)
    mlp_emb = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42).fit(X_emb, y)
    lr_emb = LogisticRegression(max_iter=1000, random_state=42).fit(X_emb, y)

    return {
        "Embeddings + Logistic Regression (Khuyên dùng - F1 cao nhất)": (emb_model, lr_emb, "embeddings"),
        "Embeddings + MLP (Mạng nơ-ron nhân tạo)": (emb_model, mlp_emb, "embeddings"),
        "Embeddings + KNN (Láng giềng gần nhất)": (emb_model, knn_emb, "embeddings"),
        "TF-IDF + SVM (Hỗ trợ vector tuyến tính)": (tfidf_vec, svm_tfidf, "tfidf"),
        "TF-IDF + Decision Tree (Cây quyết định)": (tfidf_vec, dt_tfidf, "tfidf"),
        "BoW + Naive Bayes (Xác suất phân loại)": (bow_vec, nb_bow, "bow"),
    }

def main():
    st.title("Phân loại Chủ đề Bài báo")
    st.write("Ứng dụng học máy phân loại chủ đề bài báo với 7 thuật toán đa dạng.")

    models_dict = load_and_train_models()

    # Chọn mô hình
    st.subheader("1. Cấu hình mô hình")
    chosen_model_name = st.selectbox(
        "Chọn giải thuật & phương pháp vector hóa:",
        list(models_dict.keys()),
        index=0
    )

    # Nhập văn bản
    st.subheader("2. Nhập văn bản cần dự đoán")
    default_text = "We propose a new deep learning approach for quantum particle physics and diphoton cross sections."
    user_input = st.text_area("Văn bản đầu vào (tiếng Anh hoặc tiếng Việt):", value=default_text, height=120)

    # Nút bấm dự đoán
    if st.button("🚀 Dự đoán nhãn (Predict)", type="primary"):
        if not user_input.strip():
            st.warning("Vui lòng nhập văn bản trước khi dự đoán.")
            return

        # Tiền xử lý
        cleaned = clean_text(user_input)
        tokenized = tokenize_text(cleaned)

        feature_extractor, model, feat_type = models_dict[chosen_model_name]

        # Vector hóa và dự đoán
        if feat_type in ("bow", "tfidf"):
            vec_input = feature_extractor.transform([tokenized])
            pred_label = model.predict(vec_input)[0]
        else: # embeddings
            emb_input = feature_extractor.encode([tokenized], show_progress_bar=False)
            pred_label = model.predict(emb_input)[0]

        st.success(f"**Kết quả dự đoán:** Nhãn `{pred_label}`")
        with st.expander("Xem chi tiết tiền xử lý"):
            st.write(f"- **Văn bản sau làm sạch & tách từ:** {tokenized}")
            st.write(f"- **Mô hình sử dụng:** {chosen_model_name}")

    # Xem biểu đồ phân tích
    st.divider()
    st.subheader("📊 Xem biểu đồ kết quả thực nghiệm")
    chart_choice = st.selectbox(
        "Chọn biểu đồ muốn xem:",
        [
            "Biểu đồ so sánh F1-Score (7 thuật toán)",
            "Ma trận nhầm lẫn (Confusion Matrix)",
            "Phân cụm 2D không gian đặc trưng (PCA & t-SNE)",
            "Phân phối nhãn (EDA)"
        ]
    )

    file_map = {
        "Biểu đồ so sánh F1-Score (7 thuật toán)": OUTPUT_DIR / "f1_comparison.png",
        "Ma trận nhầm lẫn (Confusion Matrix)": OUTPUT_DIR / "confusion_matrix.png",
        "Phân cụm 2D không gian đặc trưng (PCA & t-SNE)": OUTPUT_DIR / "clusters_2d.png",
        "Phân phối nhãn (EDA)": OUTPUT_DIR / "label_distribution.png",
    }
    target_img = file_map[chart_choice]
    if target_img.exists():
        st.image(str(target_img), use_container_width=True)
    else:
        st.info("Biểu đồ chưa được tạo. Hãy chạy file `main.py` để sinh đầy đủ biểu đồ.")

if __name__ == "__main__":
    main()