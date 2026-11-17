import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from src.vectorize import extract_bow, extract_tfidf, extract_embeddings

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

def plot_clusters_2d(X_emb: np.ndarray, y: np.ndarray, max_points: int = 1500):
    """
    Trực quan hóa không gian ngữ nghĩa 2D bằng PCA và t-SNE:
    - PCA: Bảo toàn phương sai toàn cục
    - t-SNE: Thể hiện cụm phi tuyến cục bộ
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Lấy mẫu con nếu dữ liệu quá lớn để vẽ nhanh và rõ ràng
    if len(y) > max_points:
        np.random.seed(42)
        idx = np.random.choice(len(y), size=max_points, replace=False)
        X_sample, y_sample = X_emb[idx], y[idx]
    else:
        X_sample, y_sample = X_emb, y

    # 1. Giảm chiều PCA 2D
    pca_2d = PCA(n_components=2, random_state=42).fit_transform(X_sample)

    # 2. Giảm chiều t-SNE 2D
    tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_sample)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Vẽ PCA 2D
    sns.scatterplot(
        x=pca_2d[:, 0], y=pca_2d[:, 1],
        hue=y_sample, palette="tab20", s=30, alpha=0.8,
        ax=axes[0], legend=False
    )
    axes[0].set_title("Trực quan hóa không gian đặc trưng 2D qua PCA", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Thành phần chính 1 (PC1)")
    axes[0].set_ylabel("Thành phần chính 2 (PC2)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Vẽ t-SNE 2D
    scatter = sns.scatterplot(
        x=tsne_2d[:, 0], y=tsne_2d[:, 1],
        hue=y_sample, palette="tab20", s=30, alpha=0.8,
        ax=axes[1]
    )
    axes[1].set_title("Trực quan hóa không gian đặc trưng 2D qua t-SNE", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("t-SNE Chiều 1")
    axes[1].set_ylabel("t-SNE Chiều 2")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Chủ đề bài báo", borderaxespad=0.)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "clusters_2d.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu biểu đồ phân cụm 2D (PCA & t-SNE) tại: {plot_path}")

def run_kmeans_clustering(feature_dict: dict, y: np.ndarray, n_clusters: int = 12):
    """
    Phân cụm học không giám sát K-Means:
    - Đánh giá trên 3 không gian đặc trưng (BoW, TF-IDF, Embeddings)
    - Chỉ số: Silhouette Score (nội tại), ARI (ngoại tại), NMI (lượng thông tin chung)
    """
    kmeans_records = []

    for feat_name, X in feature_dict.items():
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Silhouette score: lấy mẫu con tối đa 3000 để tính toán nhanh
        n_samples = X.shape[0]
        if n_samples > 3000:
            np.random.seed(42)
            sample_idx = np.random.choice(n_samples, size=3000, replace=False)
            sil_score = silhouette_score(X[sample_idx], cluster_labels[sample_idx])
        else:
            sil_score = silhouette_score(X, cluster_labels)

        ari = adjusted_rand_score(y, cluster_labels)
        nmi = normalized_mutual_info_score(y, cluster_labels)

        kmeans_records.append({
            "Feature": feat_name,
            "Clusters": n_clusters,
            "Silhouette_Score": sil_score,
            "Adjusted_Rand_Index": ari,
            "Normalized_Mutual_Info": nmi
        })

    df_kmeans = pd.DataFrame(kmeans_records)
    df_kmeans.to_csv(OUTPUT_DIR / "kmeans_results.csv", index=False)
    df_kmeans.to_latex(OUTPUT_DIR / "kmeans_table.tex", index=False, float_format="%.4f")
    print(f"Đã lưu kết quả K-Means tại: {OUTPUT_DIR / 'kmeans_results.csv'} và {OUTPUT_DIR / 'kmeans_table.tex'}")

    # Vẽ biểu đồ so sánh các chỉ số K-Means
    plt.figure(figsize=(10, 6))
    df_melt = pd.melt(
        df_kmeans,
        id_vars=["Feature"],
        value_vars=["Silhouette_Score", "Adjusted_Rand_Index", "Normalized_Mutual_Info"],
        var_name="Metric",
        value_name="Score"
    )
    metric_vn = {
        "Silhouette_Score": "Silhouette Score",
        "Adjusted_Rand_Index": "Adjusted Rand Index (ARI)",
        "Normalized_Mutual_Info": "Normalized Mutual Info (NMI)"
    }
    df_melt["Metric"] = df_melt["Metric"].map(metric_vn)

    sns.barplot(data=df_melt, x="Feature", y="Score", hue="Metric", palette="Set2")
    plt.title(f"Đánh giá thuật toán phân cụm không giám sát K-Means ({n_clusters} cụm)", fontsize=12, fontweight="bold")
    plt.ylabel("Điểm đánh giá (Score)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "kmeans_clustering.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ K-Means: {plot_path}")
    return df_kmeans

def get_models(feature_type: str):
    """
    Khởi tạo danh sách 7 giải thuật Machine Learning theo nguyên tắc ghép cặp chuẩn:
    - Với BoW & TF-IDF (Sparse): Dùng MultinomialNB. TUYỆT ĐỐI KHÔNG dùng GaussianNB (làm dày ma trận gây tràn RAM).
    - Với Sentence Embeddings (Dense): Dùng GaussianNB. TUYỆT ĐỐI KHÔNG dùng MultinomialNB (do vector chứa giá trị âm).
    - Khoảng cách KNN: Dùng khoảng cách 'cosine' cho Embeddings và 'minkowski' cho BoW/TF-IDF.
    """
    if feature_type == "Embeddings":
        # Vector dày (Dense) chứa cả giá trị âm -> GaussianNB là chuẩn tắc
        nb_model = GaussianNB()
        knn_metric = "cosine"
    else:
        # Ma trận thưa (Sparse) chứa tần suất / trọng số không âm -> MultinomialNB
        nb_model = MultinomialNB()
        knn_metric = "minkowski"

    return {
        "Naive Bayes": nb_model,
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": LinearSVC(dual="auto", random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric=knn_metric),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    }

def run_experiment(df: pd.DataFrame, top_k_classes: int = 12, test_size: float = 0.2):
    """
    Huấn luyện và đánh giá 7 mô hình trên 3 bộ vector đặc trưng (BoW, TF-IDF, Embeddings):
    - Chia dữ liệu theo tỉ lệ 80/20 có Stratify
    - Tính toán các chỉ số: Accuracy, Precision, Recall, F1 (Macro & Weighted)
    - Xuất biểu đồ so sánh F1, Confusion Matrix, phân cụm 2D và kết quả K-Means
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Lọc top K nhãn phổ biến nhất để đảm bảo stratify cân đối
    top_labels = df["label"].value_counts().nlargest(top_k_classes).index
    df_filtered = df[df["label"].isin(top_labels)].copy().reset_index(drop=True)
    texts = df_filtered["cleaned_text"].astype(str).tolist()
    y = np.array(df_filtered["label"].tolist())

    print(f"Tổng mẫu sau khi lọc Top {top_k_classes} nhãn: {len(df_filtered)}")

    # 2. Trích xuất 3 bộ đặc trưng độc lập
    print("Đang trích xuất đặc trưng BoW, TF-IDF và Embeddings...")
    X_bow, _ = extract_bow(texts)
    X_tfidf, _ = extract_tfidf(texts)
    X_emb, _ = extract_embeddings(texts)

    # 3. Trực quan hóa phân cụm 2D (PCA & t-SNE)
    plot_clusters_2d(X_emb, y)

    feature_dict = {
        "BoW": X_bow,
        "TF-IDF": X_tfidf,
        "Embeddings": X_emb,
    }

    # 4. Phân cụm học không giám sát K-Means
    print(f"\nĐang thực hiện phân cụm học không giám sát K-Means ({top_k_classes} cụm)...")
    df_kmeans = run_kmeans_clustering(feature_dict, y, n_clusters=top_k_classes)

    # 5. Huấn luyện và đánh giá các mô hình có giám sát
    results = []
    best_f1 = -1.0
    best_cm_info = None

    for feat_name, X in feature_dict.items():
        # Stratified Train/Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        models = get_models(feat_name)
        for model_name, model in models.items():
            # Huấn luyện mô hình
            model.fit(X_train, y_train)

            # Dự đoán trên tập kiểm thử
            y_pred = model.predict(X_test)

            # Đánh giá đa chiều
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
            prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)

            results.append({
                "Feature": feat_name,
                "Model": model_name,
                "Accuracy": acc,
                "Precision_Macro": prec_macro,
                "Recall_Macro": rec_macro,
                "F1_Macro": f1_macro,
                "Precision_Weighted": prec_weighted,
                "Recall_Weighted": rec_weighted,
                "F1_Weighted": f1_weighted,
            })

            # Theo dõi mô hình tốt nhất để vẽ Confusion Matrix
            if f1_macro > best_f1:
                best_f1 = f1_macro
                sorted_labels = sorted(list(np.unique(y)))
                best_cm_info = {
                    "cm": confusion_matrix(y_test, y_pred, labels=sorted_labels),
                    "labels": sorted_labels,
                    "title": f"Ma trận nhầm lẫn tốt nhất: {feat_name} + {model_name} (F1-Macro: {f1_macro:.3f})"
                }

    results_df = pd.DataFrame(results)

    # 6. Xuất bảng kết quả ra CSV và LaTeX
    results_df.to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False)
    results_df.to_latex(OUTPUT_DIR / "evaluation_table.tex", index=False, float_format="%.4f")
    print(f"Đã lưu bảng kết quả: {OUTPUT_DIR / 'evaluation_results.csv'} và {OUTPUT_DIR / 'evaluation_table.tex'}")

    # 7. Vẽ biểu đồ so sánh F1-Score (Macro)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="Model", y="F1_Macro", hue="Feature", palette="viridis")
    plt.title("So sánh F1-Score (Macro) giữa các thuật toán và bộ vector đặc trưng")
    plt.ylabel("F1-Score (Macro)")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    f1_plot_path = OUTPUT_DIR / "f1_comparison.png"
    plt.savefig(f1_plot_path, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ so sánh F1: {f1_plot_path}")

    # 8. Vẽ Confusion Matrix cho mô hình đạt F1 cao nhất
    if best_cm_info:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            best_cm_info["cm"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=best_cm_info["labels"],
            yticklabels=best_cm_info["labels"]
        )
        plt.title(best_cm_info["title"])
        plt.xlabel("Dự đoán (Predicted)")
        plt.ylabel("Thực tế (Actual)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_plot_path = OUTPUT_DIR / "confusion_matrix.png"
        plt.savefig(cm_plot_path, dpi=300)
        plt.close()
        print(f"Đã lưu Confusion Matrix: {cm_plot_path}")

    return results_df, df_kmeans

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "processed_data.csv"
    if not data_path.exists():
        from src.load_data import load_data
        from src.preprocess import preprocess_dataset
        df = load_data(sample_size=7000)
        df = preprocess_dataset(df)
    else:
        df = pd.read_csv(data_path)

    results, df_kmeans = run_experiment(df, top_k_classes=12)
    print("\nKết quả Học có giám sát (7 thuật toán ML):")
    print(results[["Feature", "Model", "Accuracy", "F1_Macro", "F1_Weighted"]].to_string())
    print("\nKết quả Học không giám sát K-Means Clustering:")
    print(df_kmeans.to_string())
