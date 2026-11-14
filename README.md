# BÀI TẬP LỚN MÔN HỌC MÁY (MACHINE LEARNING)
## Đề tài: Phân loại chủ đề bài báo

Dự án triển khai quy trình hoàn chỉnh từ thu thập, khám phá dữ liệu (EDA), tiền xử lý văn bản tiếng Việt & tiếng Anh, trích xuất đặc trưng đa dạng (BoW, TF-IDF, Sentence Embeddings), đến huấn luyện và so sánh hiệu năng của 7 giải thuật Machine Learning (Naive Bayes, Logistic Regression, SVM, Decision Tree, Random Forest, KNN, MLP). Dự án có kèm giao diện trực quan xây dựng bằng Streamlit.

---

## 📁 1. Cấu trúc thư mục

```text
BTL_HocMay/
├── data/                    # Chứa tập dữ liệu thô và dữ liệu sau tiền xử lý (processed_data.csv)
├── src/                     # Mã nguồn chính theo từng module
│   ├── load_data.py         # Nạp dữ liệu từ Hugging Face Cache
│   ├── eda.py               # Khám phá dữ liệu & xuất biểu đồ phân phối nhãn
│   ├── preprocess.py        # Làm sạch văn bản & tách từ tiếng Việt (pyvi)
│   ├── vectorize.py         # Trích xuất đặc trưng (BoW, TF-IDF kết hợp N-gram + Stopwords, Sentence Embeddings)
│   └── train_eval.py        # Huấn luyện 7 thuật toán ML, vẽ biểu đồ 2D PCA/t-SNE & đánh giá chỉ số
├── outputs/                 # Lưu biểu đồ (.png) và bảng kết quả (.csv, .tex cho LaTeX)
├── main.py                  # Script chính gọi liên hoàn từ A đến Z
├── app.py                   # Giao diện Web GUI bằng Streamlit (điểm thưởng)
├── requirements.txt         # Danh sách thư viện sử dụng
└── README.md                # Báo cáo hướng dẫn chạy mã nguồn
```

---

## 🔬 2. Cách hoạt động & Quy trình xử lý

Dự án hoạt động theo pipeline 5 bước tuần tự. Mỗi bước được đóng gói thành một module riêng trong `src/`, được gọi liên hoàn bởi `main.py`.

```mermaid
flowchart LR
    A["1. Nạp dữ liệu<br/>load_data.py"] --> B["2. Khám phá dữ liệu<br/>eda.py"]
    B --> C["3. Tiền xử lý<br/>preprocess.py"]
    C --> D["4. Vector hóa<br/>vectorize.py"]
    D --> E["5. Huấn luyện & Đánh giá<br/>train_eval.py"]
```

### Bước 1: Nạp dữ liệu (`src/load_data.py`)
- **Nguồn dữ liệu:** Tập `UniverseTBD/arxiv-abstracts-large` (~2.29 triệu bài tóm tắt khoa học từ arXiv) được lưu sẵn trong thư mục `huggingface_cache/`.
- **Xử lý:** Sử dụng thư viện `datasets` của Hugging Face để nạp từ cache local (không cần kết nối mạng sau lần tải đầu tiên), hỗ trợ lấy mẫu con (`sample_size`) để tiết kiệm RAM.
- **Đầu ra:** Một `pandas.DataFrame` với các cột chính gồm `abstract` (nội dung tóm tắt) và `categories` (nhãn phân loại lĩnh vực khoa học, ví dụ: `hep-ph`, `astro-ph`, `quant-ph`).

### Bước 2: Khám phá dữ liệu — EDA (`src/eda.py`)
- **Thống kê:** Đếm tổng số mẫu, tính phân phối độ dài văn bản theo số từ (min, max, mean, std), và đếm tần suất xuất hiện của từng nhãn.
- **Trực quan hóa:** Vẽ biểu đồ cột (Bar Chart) Top 15 nhãn phổ biến nhất bằng `seaborn`/`matplotlib` → Lưu tự động ra `outputs/label_distribution.png` (300 DPI, sẵn sàng chèn vào báo cáo LaTeX).

### Bước 3: Tiền xử lý văn bản (`src/preprocess.py`)
Quy trình làm sạch mỗi văn bản qua 2 giai đoạn nối tiếp:

1. **`clean_text(text)`** — Chuẩn hóa văn bản thô:
   - Loại bỏ toàn bộ thẻ HTML (`<tag>...</tag>`)
   - Chuyển toàn bộ ký tự về chữ thường
   - Xóa chữ số, ký tự đặc biệt, dấu câu
   - Chuẩn hóa khoảng trắng thừa

2. **`tokenize_text(text)`** — Tách từ tiếng Việt:
   - Sử dụng `ViTokenizer` từ thư viện `pyvi` để ghép các từ ghép tiếng Việt bằng dấu gạch dưới (ví dụ: `trí tuệ nhân tạo` → `trí_tuệ nhân_tạo`)
   - Tương thích tốt với cả văn bản tiếng Anh

- **Nhãn:** Chuẩn hóa cột `categories` bằng cách lấy nhãn chính đầu tiên (ví dụ: `"math.CO cs.CG"` → `"math.CO"`).
- **Đầu ra:** DataFrame 2 cột (`cleaned_text`, `label`) được lưu ra file `data/processed_data.csv`.

### Bước 4: Vector hóa — Trích xuất đặc trưng (`src/vectorize.py`)
Văn bản sau khi làm sạch được chuyển thành 3 bộ vector số độc lập để các thuật toán ML có thể xử lý:

| Phương pháp | Thư viện | Cách hoạt động | Kích thước vector |
|---|---|---|---|
| **Bag-of-Words (BoW)** | `CountVectorizer` (scikit-learn) | Đếm tần suất từ vựng kết hợp **N-gram (1, 2)** và loại bỏ **Stopwords** tiếng Anh. Vector thưa (sparse). | tối đa 5000 chiều |
| **TF-IDF** | `TfidfVectorizer` (scikit-learn) | Trọng số TF-IDF kết hợp **N-gram (1, 2)** và loại bỏ **Stopwords**. Tăng trọng số cho cụm từ khóa chuyên môn. | tối đa 5000 chiều |
| **Sentence Embeddings** | `SentenceTransformer` (sentence-transformers) | Mô hình pre-trained đa ngôn ngữ `paraphrase-multilingual-MiniLM-L12-v2` mã hóa toàn bộ ngữ nghĩa câu. | 384 chiều |

### Bước 5: Huấn luyện & Đánh giá (`src/train_eval.py`)

**Phân chia dữ liệu:**
- Lọc Top 12 nhãn phổ biến nhất để đảm bảo đủ mẫu cho phân tầng.
- Chia Train/Test theo tỉ lệ 80/20 với `stratify` (đảm bảo tỉ lệ nhãn giữ nguyên ở cả train và test).

**7 thuật toán được huấn luyện trên mỗi bộ vector (tổng cộng 21 thí nghiệm):**

| Thuật toán | Mô tả ngắn | Ghi chú |
|---|---|---|
| **Naive Bayes** | Phân loại xác suất dựa trên định lý Bayes | `MultinomialNB` cho BoW/TF-IDF, `GaussianNB` cho Embeddings (vì Embeddings chứa giá trị âm) |
| **Logistic Regression** | Hồi quy logistic đa lớp | `max_iter=1000` |
| **SVM** | Máy vector hỗ trợ tuyến tính | `LinearSVC` |
| **Decision Tree** | Cây quyết định phân loại nhị phân/đa lớp | `DecisionTreeClassifier` |
| **Random Forest** | Rừng ngẫu nhiên (tập hợp nhiều cây quyết định) | `n_estimators=100` |
| **KNN** | Thuật toán láng giềng gần nhất | `n_neighbors=5`, khoảng cách cosine cho Embeddings |
| **MLP (Neural Net)** | Mạng nơ-ron nhân tạo đa tầng | 2 ẩn tầng `(128, 64)`, tối ưu bằng Adam |

**Chỉ số đánh giá:** Accuracy, Precision, Recall, F1-Score (cả Macro và Weighted).

**Biểu đồ đầu ra tự động:**
- `outputs/f1_comparison.png` — Biểu đồ cột so sánh F1-Macro của 7 thuật toán × 3 bộ đặc trưng.
- `outputs/confusion_matrix.png` — Ma trận nhầm lẫn ($12 \times 12$) của mô hình đạt F1-Macro cao nhất.
- `outputs/clusters_2d.png` — Biểu đồ phân cụm không gian 2D (PCA & t-SNE) của 12 lớp chủ đề bài báo.
- `outputs/evaluation_results.csv` & `outputs/evaluation_table.tex` — Bảng kết quả dạng CSV và LaTeX.

---

## ⚙️ 3. Hướng dẫn Cài đặt Môi trường

Khuyến nghị sử dụng Python 3.10 trở lên. Mở Terminal / PowerShell tại thư mục dự án và thực hiện:

### Bước 1: Tạo và kích hoạt môi trường ảo (tùy chọn nhưng khuyến nghị)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Bước 2: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

---

## 🚀 4. Hướng dẫn Chạy Dự án

### Cách 1: Chạy toàn bộ quy trình từ A đến Z (Khuyến nghị)
Chạy script chính để tự động thực hiện: Nạp dữ liệu $\to$ EDA $\to$ Tiền xử lý $\to$ Vector hóa $\to$ Huấn luyện & Đánh giá:
```bash
python main.py
```
Sau khi hoàn thành, toàn bộ biểu đồ và bảng kết quả sẽ được lưu vào thư mục `outputs/`:
* `outputs/label_distribution.png`: Biểu đồ phân phối nhãn.
* `outputs/f1_comparison.png`: Biểu đồ cột so sánh F1-Score giữa các phương pháp trích xuất đặc trưng.
* `outputs/confusion_matrix.png`: Ma trận nhầm lẫn của mô hình đạt F1 cao nhất.
* `outputs/clusters_2d.png`: Biểu đồ phân cụm không gian đặc trưng 2D bằng PCA và t-SNE.
* `outputs/evaluation_results.csv` & `outputs/evaluation_table.tex`: Bảng kết quả sẵn sàng chèn vào báo cáo LaTeX.

---

### Cách 2: Khởi chạy Giao diện Web Trực quan (GUI Streamlit)
Để tương tác trực tiếp, nhập văn bản tùy ý và dự đoán nhãn:
```bash
streamlit run app.py
```
Giao diện sẽ tự động mở tại trình duyệt theo địa chỉ `http://localhost:8501`. Người dùng có thể:
1. Lựa chọn thuật toán và phương pháp vector hóa.
2. Nhập văn bản bất kỳ và bấm **Dự đoán nhãn (Predict)**.
3. Xem trực quan các biểu đồ thực nghiệm (EDA, F1-Score, Confusion Matrix) ngay trên ứng dụng.

---

### Cách 3: Chạy độc lập từng Module con (Debug / Kiểm thử)
* Nạp dữ liệu:
  ```bash
  python -m src.load_data
  ```
* Phân tích khám phá dữ liệu (EDA):
  ```bash
  python -m src.eda
  ```
* Tiền xử lý & tách từ tiếng Việt:
  ```bash
  python -m src.preprocess
  ```
* Trích xuất vector đặc trưng:
  ```bash
  python -m src.vectorize
  ```
* Huấn luyện & Đánh giá mô hình:
  ```bash
  python -m src.train_eval
  ```

---

## 📊 5. Tóm tắt Kết quả Thực nghiệm

Thực nghiệm so sánh 21 tổ hợp (7 thuật toán × 3 bộ đặc trưng) trên **12 lớp chủ đề bài báo** (sau khi tích hợp **N-gram (1, 2) + lọc Stopwords**), được sắp xếp theo F1-Macro giảm dần:

| Bộ đặc trưng | Thuật toán | Accuracy | F1-Macro | F1-Weighted |
|---|---|---|---|---|
| **Embeddings** | **Logistic Regression** | **78.51%** | **0.724** | **0.782** |
| **Embeddings** | **MLP (Neural Net)** | **77.69%** | **0.708** | **0.772** |
| **Embeddings** | **SVM** | **74.38%** | **0.681** | **0.740** |
| **Embeddings** | **Naive Bayes** | **73.55%** | **0.678** | **0.736** |
| **BoW** | **Naive Bayes** | **76.03%** | **0.664** | **0.751** |
| **TF-IDF** | **KNN** | **71.90%** | **0.657** | **0.708** |
| **TF-IDF** | **MLP (Neural Net)** | **77.69%** | **0.639** | **0.754** |
| BoW | MLP (Neural Net) | 75.21% | 0.614 | 0.724 |
| TF-IDF | SVM | 73.55% | 0.607 | 0.713 |
| BoW | SVM | 69.42% | 0.603 | 0.687 |
| BoW | Logistic Regression | 71.90% | 0.582 | 0.695 |
| Embeddings | KNN | 66.94% | 0.498 | 0.643 |
| BoW | Random Forest | 60.33% | 0.362 | 0.525 |
| BoW | Decision Tree | 49.59% | 0.355 | 0.485 |
| TF-IDF | Random Forest | 57.85% | 0.352 | 0.511 |
| Embeddings | Random Forest | 56.20% | 0.336 | 0.500 |
| TF-IDF | Decision Tree | 50.41% | 0.326 | 0.480 |
| Embeddings | Decision Tree | 47.11% | 0.301 | 0.451 |
| TF-IDF | Logistic Regression | 47.93% | 0.246 | 0.394 |
| BoW | KNN | 39.67% | 0.141 | 0.285 |
| TF-IDF | Naive Bayes | 40.50% | 0.156 | 0.299 |

**Nhận xét:**
* **Hiệu quả rõ rệt của N-gram (1, 2) và Stopwords:** Việc lọc từ dừng và giữ lại các cụm từ ghép chuyên ngành (bi-grams như `quantum computing`, `machine learning`) đã giúp **BoW + Naive Bayes** tăng vọt F1-Macro từ **0.416 $\to$ 0.664** (+24.8%) và **TF-IDF + KNN** tăng từ **0.513 $\to$ 0.657** (+14.4%).
* **Sentence Embeddings** vẫn duy trì phong độ vượt trội, chiếm trọn 4 vị trí dẫn đầu, tiêu biểu là **Logistic Regression** (Accuracy: **78.51%**, F1-Macro: **0.724**) và **MLP (Mạng nơ-ron)** (Accuracy: **77.69%**, F1-Macro: **0.708**).
* **Trực quan hóa không gian 2D (PCA & t-SNE):** Biểu đồ `outputs/clusters_2d.png` thể hiện sự tách cụm rất tự nhiên giữa các ngành toán học (`math.PR`), thiên văn học (`astro-ph`) và vật lý lượng tử (`quant-ph`).
* **Ma trận nhầm lẫn ($12 \times 12$):** Phân tích chi tiết tại `outputs/confusion_matrix.png` cho thấy mô hình nhận diện chính xác gần như tuyệt đối nhóm chủ đề vũ trụ học và thiên văn.

Các kết quả chi tiết được tổng hợp tự động trong file `outputs/evaluation_results.csv` và bảng LaTeX `outputs/evaluation_table.tex`.

---

## 💡 6. Các câu ví dụ mẫu thử nghiệm trên Web GUI (app.py)

Bạn có thể copy các đoạn văn mẫu sau vào giao diện Web để kiểm thử khả năng dự đoán:

* **Ví dụ 1 (Chủ đề thiên văn / vũ trụ):**
  > *"We present observations of distant galaxies, black holes and cosmic microwave background radiation using modern space telescopes."*
  * $\to$ **Dự đoán:** `astro-ph` (Astrophysics).

* **Ví dụ 2 (Chủ đề vật lý lượng tử):**
  > *"We study quantum entanglement, qubits state teleportation and decoherence in quantum computing circuits."*
  * $\to$ **Dự đoán:** `quant-ph` (Quantum Physics).

* **Ví dụ 3 (Chủ đề hạt cơ bản / năng lượng cao):**
  > *"Calculation of prompt diphoton production cross sections at Tevatron and LHC energies using perturbative quantum chromodynamics."*
  * $\to$ **Dự đoán:** `hep-ph` (High Energy Physics - Phenomenology).
