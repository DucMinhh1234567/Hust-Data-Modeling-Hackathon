# HBAAC Auto Parts Demand Forecasting — WRMSSE Solution

Repository này dùng để tái hiện kết quả submission cuối cùng cho bài toán dự báo nhu cầu bán hàng theo SKU trong cuộc thi dự báo demand ngành phụ tùng ô tô.

## 1. Thông tin đội thi

**Tên đội:** Tứ Đại Chiến Thần

| Vai trò | Họ và tên |
|---|---|
| Nhóm trưởng | Nguyễn Thị Thùy |
| Thành viên | Nguyễn Tiến Sơn |
| Thành viên | Trần Đức Minh |
| Thành viên | Lê Minh Tuấn |

## 2. Kết quả cuối cùng

- Public score tốt nhất trước khi hiệu chỉnh: **0.48999**
- Public score sau khi áp dụng SKU correction factor: **0.48599**
- Metric: **WRMSSE**.

Pipeline cuối cùng sử dụng:

1. Làm sạch dữ liệu transaction và tạo demand target không âm: `demand_qty = max(Quantity, 0)`.
2. Tính profit weight cho từng SKU theo `SalesAmount - Cost Amount`; SKU có profit âm hoặc bằng 0 được đưa về weight 0.
3. Xây dựng baseline ổn định bằng rolling mean nhiều cửa sổ và điều chỉnh theo weekday.
4. Train LightGBM với target dạng `ratio` trên top **2,892 SKU** có profit weight cao nhất.
5. Blend LightGBM với baseline bằng `alpha = 1.0` cho nhóm top SKU.
6. Áp dụng SKU-level correction factor học từ 3 rolling validation folds:
   - candidate SKU: top 50 SKU theo contribution/weight
   - factor grid: `0.60 → 1.20`
   - aggregation: `improvement_weighted`
   - shrink strength: `0.60`
   - max selected SKUs: `15`
   - min mean improvement: `0.00005`

File submission cuối được tạo tại:

```text
outputs/submission_lgbm_ratio_sku_corr_shrink060_sel15.csv
```

## 3. Cấu trúc thư mục

```text
.
├── data/
│   ├── raw/                         # đặt train.csv và sample_submission.csv tại đây
│   └── processed/                   # dữ liệu trung gian sinh ra từ preprocessing
├── notebooks/
│   ├── 01_eda.ipynb                 # phân tích dữ liệu ban đầu
│   ├── 02_data_preprocessing.ipynb  # làm sạch và chuẩn hóa dữ liệu
│   └── 03_model_training_submission.ipynb
├── outputs/                         # validation summary và submission cuối
├── reports/
│   └── figures/                     # biểu đồ EDA nếu được export
├── src/
│   ├── config.py                    # toàn bộ cấu hình pipeline
│   ├── pipeline.py                  # các hàm xử lý dữ liệu, model, metric, submission
│   └── run_final_submission.py      # script chạy end-to-end để tái hiện kết quả
├── requirements.txt
└── README.md
```

## 4. Giải thích source code

### `notebooks/01_eda.ipynb`

Notebook này dùng để phân tích dữ liệu gốc trước khi modeling. Nội dung chính:

- Đọc `train.csv` và `sample_submission.csv`.
- Kiểm tra số dòng, số SKU, khoảng thời gian dữ liệu.
- Phân tích các cột quan trọng: `Date`, `ItemCode`, `Quantity`, `SalesAmount`, `Cost Amount`.
- Kiểm tra giao dịch có `Quantity < 0` và giải thích tại sao không dùng trực tiếp làm demand dương.
- Phân tích phân phối profit weight theo SKU để thấy tính long-tail của bài toán.
- Xác định nhóm SKU quan trọng nhất, ví dụ top 2,892 SKU chiếm phần lớn weight chấm điểm.

Notebook này không tạo submission; mục đích là hiểu dữ liệu và định hướng modeling.

### `notebooks/02_data_preprocessing.ipynb`

Notebook này chuẩn hóa dữ liệu để dùng cho mô hình. Nội dung chính:

- Parse cột ngày về định dạng `datetime`.
- Chuẩn hóa các cột số có định dạng tiền tệ/decimal nếu cần.
- Tạo demand target không âm:

```python
y = max(Quantity, 0)
```

- Tính profit cho từng dòng:

```python
profit = SalesAmount - Cost Amount
```

- Tính tổng profit theo SKU, sau đó clip SKU có profit âm về 0 khi tính weight.
- Aggregate dữ liệu transaction về dạng daily demand theo `ItemCode` và `Date`.
- Chuẩn bị các bảng trung gian cần thiết cho modeling.

### `notebooks/03_model_training_submission.ipynb`

Notebook này là notebook chính để tái hiện kết quả cuối cùng. Nội dung chính:

1. Tạo rolling validation folds với horizon 28 ngày.
2. Xây baseline forecast cho toàn bộ SKU.
3. Tạo feature cho LightGBM:
   - lag features
   - rolling mean/rolling std
   - EWMA trend features
   - date features
   - static SKU features liên quan đến profit/weight
4. Train LightGBM target mode `ratio` cho top 2,892 SKU.
5. Tính WRMSSE local theo profit weight.
6. Học SKU correction factor từ rolling validation.
7. Retrain model trên full training data.
8. Forecast 56 ngày tiếp theo.
9. Ghi submission theo format `sample_submission.csv`.

### `src/config.py`

File này chứa toàn bộ cấu hình của pipeline. Các cấu hình quan trọng:

- Đường dẫn dữ liệu và output.
- Random seed.
- Số SKU dùng để train LightGBM: `TOP_N_SKUS = 2892`.
- Cấu hình LightGBM.
- Cấu hình baseline rolling windows.
- Cấu hình rolling validation.
- Cấu hình SKU correction factor.
- Tên file submission cuối.

Khi muốn thay đổi thử nghiệm, nên chỉnh trong file này thay vì sửa trực tiếp nhiều nơi trong code.

### `src/pipeline.py`

Đây là file lõi chứa các hàm dùng lại trong notebooks và script. Các nhóm hàm chính:

- **Load data:** đọc raw data hoặc processed data.
- **Preprocessing:** xử lý ngày, quantity âm, profit, aggregate daily demand.
- **Metric:** tính RMSSE, WRMSSE, scale denominator và profit weight.
- **Baseline:** tạo dự báo baseline bằng weighted rolling mean và weekday adjustment.
- **Feature engineering:** tạo lag, rolling, EWMA, calendar và static SKU features.
- **LightGBM:** train model, predict recursive 56 ngày.
- **Blending:** kết hợp baseline và LightGBM.
- **Correction factor:** chọn và áp dụng SKU-level correction factor.
- **Submission:** convert long prediction về đúng format `id, F1, ..., F28`.

### `src/run_final_submission.py`

Script này dùng để chạy toàn bộ pipeline mà không cần mở notebook. Flow chính:

1. Load data.
2. Tạo rolling folds.
3. Train/validate LightGBM trên từng fold.
4. Học correction factors.
5. Train model cuối trên toàn bộ train set.
6. Forecast 56 ngày.
7. Apply correction factors.
8. Lưu file submission vào `outputs/`.

Có thể dùng script này để BTC tái hiện nhanh kết quả từ terminal.

## 5. Cách tái hiện kết quả

### Bước 1: Cài thư viện

```bash
pip install -r requirements.txt
```

### Bước 2: Đặt dữ liệu gốc

Do giới hạn chia sẻ dữ liệu, repo không đính kèm file data gốc. Cần đặt đúng 2 file BTC cung cấp vào thư mục sau:

```text
data/raw/train.csv
data/raw/sample_submission.csv
```

### Bước 3: Chạy notebook theo thứ tự

```text
notebooks/01_eda.ipynb
notebooks/02_data_preprocessing.ipynb
notebooks/03_model_training_submission.ipynb
```

### Bước 4: Hoặc chạy script end-to-end

Từ thư mục root của repo:

```bash
python -m src.run_final_submission
```

Kết quả sẽ được lưu tại:

```text
outputs/submission_lgbm_ratio_sku_corr_shrink060_sel15.csv
```

## 6. Mô tả phương pháp

### 6.1. Xử lý Quantity âm

Các dòng có `Quantity < 0` thường biểu diễn đổi trả, bảo hành, hàng tặng, chuyển kho hoặc hao hụt. Vì submission yêu cầu dự báo demand không âm nên target chính được tạo bằng cách chỉ lấy phần demand dương:

```python
demand_qty = max(Quantity, 0)
```

Dự báo cuối cùng luôn được clip không âm trước khi ghi submission.

### 6.2. Profit weight

Metric WRMSSE không chấm đều các SKU. Mỗi SKU được gán trọng số theo profit:

```python
profit_i = sum(SalesAmount - Cost Amount)
weight_i = max(profit_i, 0) / sum(max(profit_i, 0))
```

Vì dữ liệu có long-tail mạnh, pipeline tập trung model chính vào nhóm SKU có weight cao.

### 6.3. Baseline

Baseline dùng weighted rolling mean ở nhiều cửa sổ thời gian:

```text
14 ngày, 28 ngày, 56 ngày, 90 ngày
```

Sau đó có điều chỉnh theo weekday để phản ánh pattern theo thứ trong tuần.

### 6.4. LightGBM ratio model

LightGBM không dự báo trực tiếp quantity tuyệt đối. Model học tỷ lệ giữa actual demand và baseline:

```python
target_ratio = actual_y / (baseline_y + eps)
```

Khi predict:

```python
pred = baseline_pred * predicted_ratio
```

Cách này giúp model ổn định hơn với các SKU có scale khác nhau.

### 6.5. Rolling validation

Dùng 3 rolling validation folds, mỗi fold dự báo 28 ngày. Score được tính bằng WRMSSE local để chọn cấu hình và correction factor.

### 6.6. SKU correction factor

Sau khi có prediction từ LightGBM, pipeline học correction factor cho một số SKU có contribution lớn. Với mỗi SKU candidate, thử nhiều factor:

```text
0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20
```

Factor cuối được aggregate theo improvement-weighted và shrink về gần 1 để giảm overfit:

```python
final_factor = 1 + 0.60 * (raw_factor - 1)
final_factor = clip(final_factor, 0.75, 1.15)
```

## 7. Output chính

Sau khi chạy xong, repo sinh ra:

```text
outputs/rolling_validation_results.csv
outputs/sku_correction_summary.csv
outputs/submission_lgbm_ratio_sku_corr_shrink060_sel15.csv
```

## 8. Lưu ý

- Submission phải giữ đúng toàn bộ `id` trong `sample_submission.csv`.
- Các giá trị forecast phải không âm.
- Nếu chạy trên máy cá nhân, thời gian train có thể lâu vì pipeline tạo frame theo SKU-day cho top 2,892 SKU.
- Kết quả có thể sai khác rất nhỏ giữa môi trường chạy do phiên bản thư viện, nhưng cấu hình cuối đã được cố định trong `src/config.py`.
