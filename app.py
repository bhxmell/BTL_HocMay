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

# -------------------------------------------------------------
# 1. Cấu hình trang & Design System (UI UX Pro Max)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Phân loại Chủ đề Bài báo Khoa học | ML Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

CATEGORY_MAPPING = {
    "astro-ph": {
        "name": "Astrophysics",
        "vn": "Vật lý Thiên văn & Vũ trụ học",
        "icon": "🔭",
        "color": "#0284c7"
    },
    "hep-ph": {
        "name": "High Energy Physics - Phenomenology",
        "vn": "Vật lý Hạt Năng lượng cao (Hiện tượng luận)",
        "icon": "💥",
        "color": "#dc2626"
    },
    "hep-th": {
        "name": "High Energy Physics - Theory",
        "vn": "Vật lý Hạt Năng lượng cao (Lý thuyết dây/hạt)",
        "icon": "🌀",
        "color": "#7c3aed"
    },
    "quant-ph": {
        "name": "Quantum Physics",
        "vn": "Vật lý Lượng tử & Máy tính Lượng tử",
        "icon": "⚛️",
        "color": "#2563eb"
    },
    "gr-qc": {
        "name": "General Relativity & Quantum Cosmology",
        "vn": "Thuyết Tương đối rộng & Vũ trụ học Lượng tử",
        "icon": "🌌",
        "color": "#475569"
    },
    "cond-mat.mes-hall": {
        "name": "Mesoscale and Nanoscale Physics",
        "vn": "Vật lý Vật chất Ngưng tụ (Quy mô Meso & Nano)",
        "icon": "🔬",
        "color": "#0d9488"
    },
    "cond-mat.str-el": {
        "name": "Strongly Correlated Electrons",
        "vn": "Vật lý Điện tử Tương quan mạnh & Siêu dẫn",
        "icon": "⚡",
        "color": "#d97706"
    },
    "cond-mat.stat-mech": {
        "name": "Statistical Mechanics",
        "vn": "Cơ học Thống kê",
        "icon": "🎲",
        "color": "#ea580c"
    },
    "math.PR": {
        "name": "Mathematics - Probability",
        "vn": "Toán học: Xác suất & Quá trình Ngẫu nhiên",
        "icon": "📈",
        "color": "#16a34a"
    },
    "math.FA": {
        "name": "Mathematics - Functional Analysis",
        "vn": "Toán học: Giải tích Hàm",
        "icon": "📐",
        "color": "#059669"
    },
    "math.CO": {
        "name": "Mathematics - Combinatorics",
        "vn": "Toán học: Tổ hợp & Lý thuyết Đồ thị",
        "icon": "🧩",
        "color": "#ca8a04"
    },
    "nucl-th": {
        "name": "Nuclear Theory",
        "vn": "Vật lý Hạt nhân (Lý thuyết)",
        "icon": "☢️",
        "color": "#b91c1c"
    }
}

SAMPLE_TEXTS = {
    "astronomy": (
        "🔭 Thiên văn học (astro-ph)",
        "We present observations of distant galaxies, supermassive black holes, and cosmological red-shift surveys using modern space telescope spectroscopy."
    ),
    "quantum": (
        "⚛️ Lượng tử (quant-ph)",
        "We demonstrate fault-tolerant quantum error correction, multi-qubit entanglement fidelity, and superconducting circuit decoherence minimization."
    ),
    "particle": (
        "💥 Vật lý hạt (hep-ph)",
        "Calculation of prompt diphoton production cross sections, hadron collisions, and Higgs boson decay channels at CERN LHC collider energies."
    ),
    "math": (
        "📈 Toán xác suất (math.PR)",
        "We establish central limit theorems, martingale inequalities, and convergence rates for stochastic differential equations driven by Brownian motion."
    )
}

# -------------------------------------------------------------
# 2. Tiêm Custom CSS theo nguyên tắc UI UX Pro Max
# -------------------------------------------------------------
st.markdown("""
<style>
    /* CSS Variables */
    :root {
        --primary: #2563eb;
        --primary-hover: #1d4ed8;
        --surface: #ffffff;
        --border: #e2e8f0;
        --text-main: #0f172a;
        --text-muted: #64748b;
        --radius: 12px;
    }

    /* Container & Layout */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Typography */
    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-weight: 700;
        color: var(--text-main);
        letter-spacing: -0.02em;
    }

    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 1.6rem 2rem;
        border-radius: var(--radius);
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px -3px rgba(15, 23, 42, 0.15);
        border: 1px solid #334155;
    }
    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 1.85rem;
        margin-bottom: 0.35rem;
    }
    .hero-banner p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 0;
    }

    /* Stat Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px -2px rgba(0, 0, 0, 0.05);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        font-weight: 600;
        margin-top: 0.2rem;
    }

    /* Card Panels */
    .card-panel {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.4rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        margin-bottom: 1.2rem;
    }

    /* Prediction Result Box */
    .result-box {
        background: linear-gradient(145deg, #f0fdf4 0%, #e0f2fe 100%);
        border: 1.5px solid #86efac;
        border-radius: var(--radius);
        padding: 1.4rem;
        margin-top: 1rem;
        animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-label {
        font-size: 1.45rem;
        font-weight: 800;
        color: #065f46;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .result-desc {
        color: #1e3a8a;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 0.2rem;
    }
    .result-vn {
        color: #475569;
        font-size: 0.9rem;
    }

    /* Confidence Badge */
    .conf-badge {
        display: inline-block;
        background: #0f172a;
        color: #f8fafc;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-left: 0.5rem;
    }

    /* Sample Pills */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 3. Cache & Nạp Mô hình Huấn luyện
# -------------------------------------------------------------
@st.cache_resource(show_spinner="⏳ Đang nạp dữ liệu và khởi tạo các mô hình Machine Learning...")
def load_and_train_models():
    """Huấn luyện sẵn các mô hình trên dữ liệu đã tiền xử lý để suy luận tức thì."""
    if not DATA_PATH.exists():
        from src.load_data import load_data
        from src.preprocess import preprocess_dataset
        df = load_data(sample_size=5000)
        df = preprocess_dataset(df)
    else:
        df = pd.read_csv(DATA_PATH)

    top_labels = list(df["label"].value_counts().nlargest(12).index)
    df_filtered = df[df["label"].isin(top_labels)].copy().reset_index(drop=True)
    texts = df_filtered["cleaned_text"].astype(str).tolist()
    y = np.array(df_filtered["label"].tolist())

    # 1. Trích xuất 3 bộ vector đặc trưng
    X_bow, bow_vec = extract_bow(texts)
    X_tfidf, tfidf_vec = extract_tfidf(texts)
    X_emb, emb_model = extract_embeddings(texts)

    # 2. Huấn luyện các mô hình tối ưu
    nb_bow = MultinomialNB().fit(X_bow, y)
    lr_bow = LogisticRegression(max_iter=1000, random_state=42).fit(X_bow, y)
    dt_tfidf = DecisionTreeClassifier(random_state=42).fit(X_tfidf, y)
    svm_tfidf = LinearSVC(dual="auto", random_state=42).fit(X_tfidf, y)
    mlp_tfidf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42).fit(X_tfidf, y)
    knn_emb = KNeighborsClassifier(n_neighbors=5, metric="cosine").fit(X_emb, y)
    mlp_emb = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42).fit(X_emb, y)
    lr_emb = LogisticRegression(max_iter=1000, random_state=42).fit(X_emb, y)

    models_catalog = {
        "TF-IDF + SVM (Accuracy: 81.86% - F1: 0.747 - Cao nhất)": (tfidf_vec, svm_tfidf, "tfidf"),
        "TF-IDF + MLP Neural Net (Accuracy: 80.65% - F1: 0.739)": (tfidf_vec, mlp_tfidf, "tfidf"),
        "BoW + MLP Neural Net (Accuracy: 80.65% - F1: 0.734)": (bow_vec, lr_bow, "bow"),
        "BoW + Logistic Regression (Accuracy: 79.32% - F1: 0.722)": (bow_vec, lr_bow, "bow"),
        "BoW + Naive Bayes (Accuracy: 80.29% - F1: 0.720)": (bow_vec, nb_bow, "bow"),
        "Embeddings + Logistic Regression (Accuracy: 73.88% - F1: 0.650)": (emb_model, lr_emb, "embeddings"),
        "Embeddings + KNN (Accuracy: 73.64% - F1: 0.641)": (emb_model, knn_emb, "embeddings"),
        "Embeddings + MLP Neural Net (Accuracy: 73.16% - F1: 0.640)": (emb_model, mlp_emb, "embeddings"),
        "TF-IDF + Decision Tree (Accuracy: 57.32% - F1: 0.449)": (tfidf_vec, dt_tfidf, "tfidf"),
    }
    return models_catalog, top_labels

def get_prediction_and_probs(model, vectorizer_or_embedder, feat_type, tokenized_text, classes):
    """Tính toán nhãn dự đoán và xác suất (hoặc softmax decision score)."""
    if feat_type in ("bow", "tfidf"):
        X_in = vectorizer_or_embedder.transform([tokenized_text])
    else:
        X_in = vectorizer_or_embedder.encode([tokenized_text], show_progress_bar=False)

    pred_label = model.predict(X_in)[0]

    # Tính xác suất phân phối
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_in)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_in)
        if scores.ndim > 1:
            scores = scores[0]
        # Softmax chuyển đổi khoảng cách thành xác suất tương đối
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
    else:
        probs = None

    top_candidates = []
    confidence = None
    if probs is not None:
        model_classes = list(model.classes_)
        sorted_idx = np.argsort(probs)[::-1]
        for idx in sorted_idx[:3]:
            cls_name = model_classes[idx]
            top_candidates.append((cls_name, float(probs[idx])))
        confidence = float(np.max(probs))

    return pred_label, confidence, top_candidates

# -------------------------------------------------------------
# 4. Giao diện Người dùng (Main Application)
# -------------------------------------------------------------
def main():
    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <h1>🔬 Hệ Thống Phân Loại Chủ Đề Bài Báo Khoa Học</h1>
        <p>Ứng dụng Machine Learning đa giải thuật (Supervised & Unsupervised K-Means) trên dữ liệu văn bản arXiv</p>
    </div>
    """, unsafe_allow_html=True)

    # 4 Thẻ KPI Tóm tắt
    st.markdown("""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-val">7.000</div>
            <div class="metric-label">Mẫu dữ liệu thực nghiệm</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">12</div>
            <div class="metric-label">Chuyên ngành bài báo</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">81.86%</div>
            <div class="metric-label">Độ chính xác cao nhất (F1: 0.747)</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">0.4792</div>
            <div class="metric-label">NMI K-Means Không giám sát</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Khởi tạo session state lưu văn bản nhập
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = SAMPLE_TEXTS["astronomy"][1]

    models_catalog, top_labels = load_and_train_models()

    # 3 Tab chính
    tab_predict, tab_analytics, tab_taxonomy = st.tabs([
        "🎯 Dự đoán Trực tiếp",
        "📊 Báo cáo Thực nghiệm & Biểu đồ",
        "📚 Bảng 12 Chuyên ngành (Taxonomy)"
    ])

    # ---------------------------------------------------------
    # TAB 1: DỰ ĐOÁN TRỰC TIẾP
    # ---------------------------------------------------------
    with tab_predict:
        col_input, col_result = st.columns([1.1, 0.9], gap="large")

        with col_input:
            st.markdown("### 1. Cấu hình & Dữ liệu đầu vào")
            
            chosen_model_name = st.selectbox(
                "Chọn Giải thuật & Không gian Đặc trưng:",
                list(models_catalog.keys()),
                index=0,
                help="Các mô hình được sắp xếp theo hiệu năng F1-Macro trên 7.000 mẫu dữ liệu."
            )

            # Nút điền mẫu thử nhanh (1-Click Sample Pills)
            st.markdown("<small style='color: #64748b; font-weight: 600;'>💡 Chèn văn bản mẫu thử nhanh:</small>", unsafe_allow_html=True)
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            if col_b1.button("🔭 Thiên văn", use_container_width=True):
                st.session_state["input_text"] = SAMPLE_TEXTS["astronomy"][1]
            if col_b2.button("⚛️ Lượng tử", use_container_width=True):
                st.session_state["input_text"] = SAMPLE_TEXTS["quantum"][1]
            if col_b3.button("💥 Vật lý hạt", use_container_width=True):
                st.session_state["input_text"] = SAMPLE_TEXTS["particle"][1]
            if col_b4.button("📈 Xác suất", use_container_width=True):
                st.session_state["input_text"] = SAMPLE_TEXTS["math"][1]

            user_text = st.text_area(
                "Đoạn tóm tắt bài báo (Abstract):",
                value=st.session_state["input_text"],
                height=150,
                placeholder="Nhập nội dung tóm tắt bài báo khoa học..."
            )
            st.session_state["input_text"] = user_text

            btn_predict = st.button("🚀 Dự đoán Chuyên ngành", type="primary", use_container_width=True)

        with col_result:
            st.markdown("### 2. Kết quả Phân loại")

            if btn_predict:
                if not user_text.strip():
                    st.warning("⚠️ Vui lòng nhập nội dung văn bản trước khi thực hiện dự đoán.")
                else:
                    with st.spinner("Đang tiền xử lý & vector hóa..."):
                        # Tiền xử lý
                        cleaned = clean_text(user_text)
                        tokenized = tokenize_text(cleaned)

                        vec_or_emb, model, feat_type = models_catalog[chosen_model_name]
                        pred_label, conf, top3 = get_prediction_and_probs(
                            model, vec_or_emb, feat_type, tokenized, top_labels
                        )

                        info = CATEGORY_MAPPING.get(pred_label, {
                            "name": pred_label,
                            "vn": "Chuyên ngành khoa học",
                            "icon": "📄",
                            "color": "#2563eb"
                        })

                        conf_pct = f"{conf * 100:.1f}%" if conf else "N/A"

                        # Card kết quả nổi bật
                        st.markdown(f"""
                        <div class="result-box">
                            <div class="result-label">
                                <span>{info['icon']} {pred_label}</span>
                                <span class="conf-badge">Độ tin cậy: {conf_pct}</span>
                            </div>
                            <div class="result-desc">{info['name']}</div>
                            <div class="result-vn">({info['vn']})</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Top 3 khả năng cao nhất
                        if top3:
                            st.markdown("<p style='font-weight: 700; margin-top: 1.2rem; margin-bottom: 0.5rem;'>Phân phối xác suất (Top 3):</p>", unsafe_allow_html=True)
                            for rank, (cand_lbl, cand_p) in enumerate(top3, 1):
                                cand_info = CATEGORY_MAPPING.get(cand_lbl, {"name": cand_lbl, "icon": "•"})
                                col_lbl, col_bar = st.columns([1, 1.2])
                                with col_lbl:
                                    st.write(f"**#{rank} {cand_info['icon']} `{cand_lbl}`**")
                                    st.caption(cand_info['name'])
                                with col_bar:
                                    st.progress(float(cand_p))
                                    st.caption(f"{cand_p*100:.1f}%")

                        # Chi tiết kiểm thử
                        with st.expander("🔍 Kiểm tra tiền xử lý (Preprocessing Inspector)"):
                            st.markdown(f"**Văn bản sau làm sạch & tách từ:**\n```text\n{tokenized}\n```")
                            st.markdown(f"- **Mô hình:** {chosen_model_name}")
                            st.markdown(f"- **Kiểu vector:** `{feat_type.upper()}`")
            else:
                st.info("👈 Bấm **'🚀 Dự đoán Chuyên ngành'** ở cột bên trái để xem kết quả phân loại và thanh đo độ tin cậy.")

    # ---------------------------------------------------------
    # TAB 2: BÁO CÁO THỰC NGHIỆM & BIỂU ĐỒ
    # ---------------------------------------------------------
    with tab_analytics:
        st.markdown("### 📊 Tổng hợp Kết quả Thực nghiệm & Biểu đồ Trực quan")
        st.write("Hệ thống tự động sinh biểu đồ phân tích 300 DPI và bảng dữ liệu sẵn sàng xuất bản báo cáo LaTeX.")

        chart_choice = st.selectbox(
            "Chọn biểu đồ phân tích thực nghiệm:",
            [
                "Biểu đồ so sánh F1-Score (7 thuật toán x 3 bộ đặc trưng)",
                "Ma trận nhầm lẫn (Confusion Matrix 12x12)",
                "Phân cụm không gian đặc trưng 2D (PCA & t-SNE)",
                "Đánh giá Phân cụm K-Means Không giám sát (Silhouette, ARI, NMI)",
                "Phân phối số lượng nhãn bài báo (EDA)"
            ]
        )

        file_map = {
            "Biểu đồ so sánh F1-Score (7 thuật toán x 3 bộ đặc trưng)": (
                OUTPUT_DIR / "f1_comparison.png",
                "Biểu đồ so sánh điểm F1-Score (Macro) của 21 cấu hình thực nghiệm trên 4.131 mẫu thử nghiệm."
            ),
            "Ma trận nhầm lẫn (Confusion Matrix 12x12)": (
                OUTPUT_DIR / "confusion_matrix.png",
                "Ma trận nhầm lẫn chi tiết giữa nhãn thực tế và nhãn dự đoán của mô hình đạt hiệu năng cao nhất."
            ),
            "Phân cụm không gian đặc trưng 2D (PCA & t-SNE)": (
                OUTPUT_DIR / "clusters_2d.png",
                "Trực quan hóa giảm chiều không gian từ 384 chiều xuống 2D thể hiện sự tách cụm tự nhiên giữa các chủ đề."
            ),
            "Đánh giá Phân cụm K-Means Không giám sát (Silhouette, ARI, NMI)": (
                OUTPUT_DIR / "kmeans_clustering.png",
                "So sánh 3 chỉ số Silhouette Score, Adjusted Rand Index (ARI), và Normalized Mutual Info (NMI) cho thuật toán K-Means."
            ),
            "Phân phối số lượng nhãn bài báo (EDA)": (
                OUTPUT_DIR / "label_distribution.png",
                "Phân phối tần suất xuất hiện của các nhãn chủ đề khoa học trên tập mẫu arXiv."
            )
        }

        target_img, img_desc = file_map[chart_choice]
        if target_img.exists():
            st.image(str(target_img), caption=img_desc, use_container_width=True)
        else:
            st.warning("⚠️ Biểu đồ chưa được tạo. Hãy chạy file `main.py` để sinh đầy đủ toàn bộ biểu đồ.")

        # Tải bảng số liệu CSV & LaTeX
        st.divider()
        st.markdown("#### 📥 Tải dữ liệu bảng thực nghiệm")
        col_d1, col_d2 = st.columns(2)
        csv_eval = OUTPUT_DIR / "evaluation_results.csv"
        tex_eval = OUTPUT_DIR / "evaluation_table.tex"
        if csv_eval.exists():
            with open(csv_eval, "rb") as f:
                col_d1.download_button(
                    label="📄 Tải Bảng Đánh giá Có giám sát (.CSV)",
                    data=f,
                    file_name="evaluation_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        if tex_eval.exists():
            with open(tex_eval, "rb") as f:
                col_d2.download_button(
                    label="📑 Tải Bảng Mã nguồn LaTeX (.TEX)",
                    data=f,
                    file_name="evaluation_table.tex",
                    mime="text/plain",
                    use_container_width=True
                )

    # ---------------------------------------------------------
    # TAB 3: BẢNG TRA CỨU 12 CHUYÊN NGÀNH
    # ---------------------------------------------------------
    with tab_taxonomy:
        st.markdown("### 📚 Danh mục Ánh xạ 12 Chuyên ngành Khoa học")
        st.write("Bảng tra cứu mã định danh arXiv sang tên tiếng Anh chính thức và bản dịch thuật ngữ tiếng Việt:")

        table_rows = []
        for code, meta in CATEGORY_MAPPING.items():
            table_rows.append({
                "Mã nhãn (arXiv)": f"{meta['icon']} `{code}`",
                "Tên Tiếng Anh": meta["name"],
                "Lĩnh vực chuyên môn (Tiếng Việt)": meta["vn"]
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
