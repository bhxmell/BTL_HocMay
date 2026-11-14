from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.vectorize import extract_bow, extract_tfidf, extract_embeddings

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

def plot_clusters_2d(X, y, output_filename: str = "clusters_2d.png"):
    """Chiếu vector Embeddings xuống không gian 2D bằng PCA và t-SNE để trực quan hóa cụm."""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    unique_labels = np.unique(y)
    palette = sns.color_palette("tab20", len(unique_labels))

    # 1. PCA Subplot
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette=palette,
        alpha=0.75, s=45, ax=axes[0], legend=False
    )
    axes[0].set_title(
        f"Trực quan hóa PCA 2D (Tỷ lệ phương sai: {pca.explained_variance_ratio_.sum()*100:.1f}%)",
        fontsize=12, fontweight="bold"
    )
    axes[0].set_xlabel("PCA Component 1")
    axes[0].set_ylabel("PCA Component 2")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. t-SNE Subplot
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette=palette,
        alpha=0.75, s=45, ax=axes[1]
    )
    axes[1].set_title(
        "Trực quan hóa phân cụm phi tuyến tính với t-SNE",
        fontsize=12, fontweight="bold"
    )
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Chủ đề bài báo")

    plt.tight_layout()
    out_path = OUTPUT_DIR / output_filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu biểu đồ phân cụm 2D (PCA & t-SNE) tại: {out_path}")

def run_kmeans_clustering(feature_dict, y, n_clusters=12):
    """Thực nghiệm phân cụm học không giám sát bằng K-Means trên 3 bộ đặc trưng."""
    kmeans_results = []
    for feat_name, X in feature_dict.items():
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        sil = silhouette_score(X, cluster_labels)
        ari = adjusted_rand_score(y, cluster_labels)
        nmi = normalized_mutual_info_score(y, cluster_labels)

        kmeans_results.append({
            "Feature": feat_name,
            "Clusters": n_clusters,
            "Silhouette_Score": sil,
            "Adjusted_Rand_Index": ari,
            "Normalized_Mutual_Info": nmi,
        })

    df_kmeans = pd.DataFrame(kmeans_results)
    csv_path = OUTPUT_DIR / "kmeans_results.csv"
    df_kmeans.to_csv(csv_path, index=False)
    tex_path = OUTPUT_DIR / "kmeans_table.tex"
    df_kmeans.to_latex(tex_path, index=False, float_format="%.4f")
    print(f"Đã lưu kết quả K-Means tại: {csv_path} và {tex_path}")

    # Vẽ biểu đồ so sánh các chỉ số phân cụm K-Means
    plt.figure(figsize=(9, 5))
    df_melt = df_kmeans.melt(
        id_vars=["Feature", "Clusters"],
        value_vars=["Silhouette_Score", "Adjusted_Rand_Index", "Normalized_Mutual_Info"],
        var_name="Metric",
        value_name="Score"
    )
    sns.barplot(data=df_melt, x="Feature", y="Score", hue="Metric", palette="Set1")
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
    """Khởi tạo danh sách 7 thuật toán ML (chọn Naive Bayes phù hợp cho từng loại đặc trưng)."""
    nb_model = GaussianNB() if feature_type == "Embeddings" else MultinomialNB()
    return {
        "Naive Bayes": nb_model,
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": LinearSVC(dual="auto", random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="cosine" if feature_type == "Embeddings" else "minkowski"),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    }

def run_experiment(df: pd.DataFrame, top_k_classes: int = 12, test_size: float = 0.2):
    """Huấn luyện và đánh giá 7 mô hình trên 3 bộ vector đặc trưng (BoW, TF-IDF, Embeddings)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Lọc top K nhãn phổ biến nhất để đảm bảo stratify cân đối
    top_labels = df["label"].value_counts().nlargest(top_k_classes).index
    df_filtered = df[df["label"].isin(top_labels)].copy().reset_index(drop=True)
    texts = df_filtered["cleaned_text"].astype(str).tolist()
    y = np.array(df_filtered["label"].tolist())

    print(f"Tổng mẫu sau khi lọc Top {top_k_classes} nhãn: {len(df_filtered)}")

    # 1. Trích xuất 3 bộ đặc trưng độc lập
    print("Đang trích xuất đặc trưng BoW, TF-IDF và Embeddings...")
    X_bow, _ = extract_bow(texts)
    X_tfidf, _ = extract_tfidf(texts)
    X_emb, _ = extract_embeddings(texts)

    # 2. Trực quan hóa phân cụm 2D (PCA & t-SNE)
    plot_clusters_2d(X_emb, y)

    feature_dict = {
        "BoW": X_bow,
        "TF-IDF": X_tfidf,
        "Embeddings": X_emb,
    }

    # 3. Phân cụm học không giám sát K-Means
    print(f"\nĐang thực hiện phân cụm học không giám sát K-Means ({top_k_classes} cụm)...")
    df_kmeans = run_kmeans_clustering(feature_dict, y, n_clusters=top_k_classes)

    results = []
    best_f1 = -1.0
    best_cm_info = None

    for feat_name, X in feature_dict.items():
        # Stratified Train/Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        models = get_models(feat_name)
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
            p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

            results.append({
                "Feature": feat_name,
                "Model": model_name,
                "Accuracy": acc,
                "Precision_Macro": p_macro,
                "Recall_Macro": r_macro,
                "F1_Macro": f1_macro,
                "Precision_Weighted": p_weighted,
                "Recall_Weighted": r_weighted,
                "F1_Weighted": f1_weighted,
            })

            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_cm_info = {
                    "cm": confusion_matrix(y_test, y_pred, labels=np.unique(y)),
                    "labels": np.unique(y),
                    "title": f"Confusion Matrix: {model_name} + {feat_name} (F1-Macro: {f1_macro:.3f})"
                }

    results_df = pd.DataFrame(results)

    # 2. Lưu bảng kết quả CSV và LaTeX
    csv_path = OUTPUT_DIR / "evaluation_results.csv"
    results_df.to_csv(csv_path, index=False)
    tex_path = OUTPUT_DIR / "evaluation_table.tex"
    results_df.to_latex(tex_path, index=False, float_format="%.4f")
    print(f"Đã lưu bảng kết quả: {csv_path} và {tex_path}")

    # 3. Vẽ biểu đồ so sánh F1-Score
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="Model", y="F1_Macro", hue="Feature", palette="Set2")
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

    # 4. Vẽ Confusion Matrix cho mô hình tốt nhất
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
