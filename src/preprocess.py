import re
from pathlib import Path
import pandas as pd
from pyvi import ViTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def clean_text(text: str) -> str:
    """Làm sạch văn bản: chuyển chữ thường, xóa HTML, ký tự đặc biệt, dấu câu, chữ số và khoảng trắng thừa."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()

def tokenize_text(text: str) -> str:
    """Tách từ tiếng Việt sử dụng pyvi."""
    return ViTokenizer.tokenize(text) if text else ""

def preprocess_dataset(
    df: pd.DataFrame,
    text_col: str = "abstract",
    label_col: str = "categories",
    output_filename: str = "processed_data.csv"
) -> pd.DataFrame:
    """Làm sạch, tách từ và lưu tập dữ liệu vào data/processed_data.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_clean = pd.DataFrame()
    df_clean["cleaned_text"] = df[text_col].apply(clean_text).apply(tokenize_text)
    if label_col in df.columns:
        df_clean["label"] = df[label_col].astype(str).apply(lambda x: x.split()[0])

    output_path = DATA_DIR / output_filename
    df_clean.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Đã lưu dữ liệu tiền xử lý tại: {output_path} ({len(df_clean)} mẫu)")
    return df_clean

if __name__ == "__main__":
    # 1. Tự kiểm tra hàm clean_text & tách từ tiếng Việt
    sample = "<p>Chào bạn, đây là thử nghiệm số 123! Trí tuệ nhân tạo.</p>"
    cleaned = clean_text(sample)
    tokenized = tokenize_text(cleaned)
    assert "123" not in tokenized and "<p>" not in tokenized and "trí_tuệ" in tokenized
    print(f"Kiểm thử làm sạch & tách từ thành công:\nGốc: {sample}\nKết quả: {tokenized}")

    # 2. Xử lý mẫu từ dữ liệu thực tế
    from src.load_data import load_data
    df = load_data(sample_size=7000)
    df_clean = preprocess_dataset(df)
    assert (DATA_DIR / "processed_data.csv").exists()
