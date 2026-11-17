import re
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def clean_text(text: str) -> str:
    """
    Làm sạch văn bản tiếng Anh:
    - Loại bỏ thẻ HTML (<tag>...</tag>)
    - Chuyển chữ thường (lowercase)
    - Loại bỏ ký tự xuống dòng (\n), tab (\t), ký tự đặc biệt, dấu câu
    - GIỮ LẠI chữ số (digits 0-9) để bảo toàn thông tin phiên bản mô hình, năm, hằng số (vd: GPT-4, BERT-2)
    - Chuẩn hóa khoảng trắng thừa
    """
    if not isinstance(text, str):
        return ""
    # 1. Loại bỏ thẻ HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # 2. Chuyển về chữ thường
    text = text.lower()
    # 3. Xóa các ký tự xuống dòng và tab
    text = re.sub(r"[\r\n\t]+", " ", text)
    # 4. Xóa ký tự đặc biệt và dấu câu, GIỮ LẠI chữ cái (a-z) và chữ số (0-9)
    # \w khớp với [a-zA-Z0-9_], thay dấu gạch dưới '_' bằng khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text).replace("_", " ")
    # 5. Chuẩn hóa khoảng trắng liên tiếp
    return re.sub(r"\s+", " ", text).strip()

def tokenize_text(text: str) -> str:
    """
    Hàm tương thích ngược (không dùng pyvi vì dữ liệu 100% tiếng Anh).
    Vector hóa tiếng Anh được xử lý trực tiếp bởi Count/TF-IDF Vectorizer và SentenceTransformer.
    """
    return text if text else ""

def preprocess_dataset(
    df: pd.DataFrame,
    title_col: str = "title",
    abstract_col: str = "abstract",
    label_col: str = "categories",
    output_filename: str = "processed_data.csv"
) -> pd.DataFrame:
    """
    Gộp tiêu đề (title) và tóm tắt (abstract), làm sạch và lưu tập dữ liệu:
    - df['text'] = df['title'].fillna('') + '. ' + df['abstract'].fillna('')
    - Lưu DataFrame gồm (cleaned_text, label) ra data/processed_data.csv
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Gộp title và abstract tạo văn bản đầu vào duy nhất
    title_series = df[title_col].fillna("") if title_col in df.columns else pd.Series([""] * len(df))
    abstract_series = df[abstract_col].fillna("") if abstract_col in df.columns else pd.Series([""] * len(df))
    combined_text = title_series.astype(str) + ". " + abstract_series.astype(str)

    # 2. Làm sạch văn bản (giữ chữ số, bỏ HTML & ký tự đặc biệt)
    df_clean = pd.DataFrame()
    df_clean["cleaned_text"] = combined_text.apply(clean_text)

    # 3. Trích xuất nhãn phân loại chính đầu tiên
    if label_col in df.columns:
        df_clean["label"] = df[label_col].astype(str).apply(lambda x: x.split()[0] if x else "unknown")

    # 4. Lưu ra file CSV
    output_path = DATA_DIR / output_filename
    df_clean.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Đã lưu dữ liệu tiền xử lý tại: {output_path} ({len(df_clean)} mẫu)")
    return df_clean

if __name__ == "__main__":
    # Tự kiểm thử hàm làm sạch với số và ký tự đặc biệt
    sample = "<p>GPT-4 and BERT-2 in 3D Quantum Gravity: arXiv:2024.12345\n\tresults!</p>"
    cleaned = clean_text(sample)
    print(f"Kiểm thử clean_text:\nGốc: {sample}\nKết quả: {cleaned}")
    assert "gpt 4" in cleaned and "bert 2" in cleaned and "3d" in cleaned and "<p>" not in cleaned
    print("-> Kiểm thử thành công! Chữ số được bảo toàn nguyên vẹn.")
