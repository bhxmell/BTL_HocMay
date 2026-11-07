from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

def explore_data(df: pd.DataFrame, text_col: str = "abstract", label_col: str = "categories", top_k: int = 15):
    """Thống kê độ dài văn bản, kiểm tra phân phối nhãn và xuất biểu đồ."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Thống kê độ dài văn bản (số từ)
    word_counts = df[text_col].dropna().astype(str).str.split().apply(len)
    print(f"Tổng số mẫu: {len(df)}")
    print(f"Thống kê số từ:\n{word_counts.describe()}\n")

    # 2. Phân phối nhãn (lấy nhãn chính đầu tiên đối với dữ liệu đa nhãn)
    primary_labels = df[label_col].dropna().astype(str).apply(lambda x: x.split()[0])
    label_counts = primary_labels.value_counts()
    print(f"Tổng số nhãn duy nhất: {len(label_counts)}")
    print(f"Top 5 nhãn xuất hiện nhiều nhất:\n{label_counts.head(5)}\n")

    # 3. Vẽ biểu đồ Bar Chart phân phối nhãn
    plt.figure(figsize=(10, 6))
    top_labels = label_counts.head(top_k)
    sns.barplot(x=top_labels.values, y=top_labels.index, hue=top_labels.index, palette="viridis", legend=False)
    plt.title(f"Top {top_k} phân phối nhãn ({label_col})")
    plt.xlabel("Số lượng mẫu")
    plt.ylabel("Nhãn")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "label_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ: {out_path}")

    return word_counts, label_counts

if __name__ == "__main__":
    from src.load_data import load_data
    df = load_data(sample_size=10000)
    explore_data(df)
