import sys
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.load_data import load_data
from src.eda import explore_data
from src.preprocess import preprocess_dataset
from src.train_eval import run_experiment

def main():
    print("=" * 60)
    print("QUY TRÌNH HỌC MÁY PHÂN LOẠI CHỦ ĐỀ BÀI BÁO TỪ A ĐẾN Z")
    print("=" * 60)

    # 1. Nạp dữ liệu
    print("\n[Bước 1/5] Nạp dữ liệu từ cache Hugging Face...")
    df_raw = load_data(sample_size=5000)
    print(f"-> Đã nạp {len(df_raw)} mẫu.")

    # 2. Khám phá & Trực quan hóa dữ liệu (EDA)
    print("\n[Bước 2/5] Khám phá dữ liệu (EDA) & xuất biểu đồ phân phối nhãn...")
    explore_data(df_raw)

    # 3. Tiền xử lý văn bản tiếng Anh (Gộp Title + Abstract, giữ chữ số)
    print("\n[Bước 3/5] Tiền xử lý: Gộp Title + Abstract, làm sạch văn bản (giữ chữ số)...")
    df_clean = preprocess_dataset(df_raw)

    # 4 & 5. Vector hóa (BoW, TF-IDF, Embeddings) và Huấn luyện & Đánh giá mô hình
    print("\n[Bước 4 & 5] Vector hóa dữ liệu, huấn luyện 7 thuật toán Học có giám sát & Phân cụm K-Means không giám sát...")
    results, df_kmeans = run_experiment(df_clean, top_k_classes=12)

    print("\n" + "=" * 60)
    print("HOÀN THÀNH TOÀN BỘ QUY TRÌNH!")
    print("=" * 60)
    print("1. Bảng tóm tắt kết quả Học có giám sát (Supervised):")
    print(results[["Feature", "Model", "Accuracy", "F1_Macro", "F1_Weighted"]].to_string())
    print("\n2. Bảng tóm tắt kết quả Phân cụm K-Means không giám sát (Unsupervised):")
    print(df_kmeans.to_string())

if __name__ == "__main__":
    main()
