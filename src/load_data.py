import os
from pathlib import Path
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "huggingface_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset

def load_data(dataset_name: str = "UniverseTBD/arxiv-abstracts-large", split: str = "train", sample_size: int | None = None) -> pd.DataFrame:
    """Nạp dataset từ cache Hugging Face và chuyển thành Pandas DataFrame."""
    ds = load_dataset(dataset_name, split=split)
    if sample_size and sample_size < len(ds):
        ds = ds.select(range(sample_size))
    return ds.to_pandas()

if __name__ == "__main__":
    df = load_data(sample_size=100)
    assert not df.empty and "abstract" in df.columns
    print(f"Nạp thành công {len(df)} mẫu. Các cột: {list(df.columns)}")
