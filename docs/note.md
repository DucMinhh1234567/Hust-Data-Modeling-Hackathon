# Phân tích data:
- Mục tiêu của bài toán là dự báo số lượng bán hàng theo ngày cho từng SKU trong 56 ngày tới (chia 2 window 28 ngày).
- Dữ liệu gồm có:
  - `train.csv`: Dữ liệu giao dịch chi tiết, 711,980 dòng, 2020-11-17 → 2025-09-05. Mỗi dòng là một giao dịch.
  - `sample_submission.csv`: Template submission (31,944 dòng × 29 cột): mỗi SKU có hai dòng `<SKU>_validation` (Public window) và `<SKU>_evaluation` (Private window), mỗi dòng có 28 cột `F1..F28`. Tất cả zeros là placeholder.
- Các cột trong `train.csv`:
  - `Date`: Ngày giao dịch.
  - `ItemCode`: Mã SKU.
  - `Quantity`: Số lượng bán hàng.
  - `UnitPrice`: Giá bán đơn vị.
  - `SalesAmount`: Doanh thu bán hàng.
  - `Unit Cost`: Giá nhập đơn vị.
  - `Cost Amount`: Chi phí nhập hàng.

## EDA:

=== QUANTITY STATS ===
count    711980.000000
mean          3.437250
std          25.490722
min        -998.000000
25%           1.000000
50%           1.000000
75%           2.000000
max        5998.000000
Name: Quantity, dtype: float64
Negative (returns): 37434
Zero quantity rows: 3164
→ Số lượng bán hàng trung bình là 3.437250, độ lệch chuẩn là 25.490722. Có thể thấy số lượng bán hàng có phân phối lệch phải, với nhiều giá trị là 1. Có 37434 dòng có số lượng bán hàng âm, có thể là các dòng trả hàng.


=== SKU COUNT ===
Unique SKUs in train: 15972

=== DAILY SALES STATS PER SKU ===
SKUs with positive total sales: 15432
SKUs with zero total sales: 537
SKUs with negative total sales: 3

=== SPARSITY ===
Total training days: 1411
Median active days per SKU: 6.0
SKUs with < 10 active days: 9259
SKUs with < 50 active days: 13631
-> Anh hưởng điểm chủ yếu qua weight profit, không phải số ngày active. SKU sparse nhưng profit cao vẫn rất quan trọng.

=== PROFIT DISTRIBUTION ===
SKUs with positive profit: 10757
SKUs with zero profit: 528
SKUs with negative profit (weight=0): 4687
Top 10 SKUs by profit:
ItemCode
SKU-00003    1.670068e+10
SKU-00002    8.012686e+09
SKU-00005    2.243340e+09
SKU-12534    1.215804e+09
SKU-08589    1.146484e+09
SKU-12537    1.106800e+09
SKU-00324    8.697472e+08
SKU-04268    8.645621e+08
SKU-06780    8.535920e+08
SKU-00004    8.359968e+08
dtype: float64

Top 99 SKUs account for 50% of profit *(EDA nhanh — xem bảng dưới)*
Top 653 SKUs account for 80% of profit *(EDA nhanh)*

### Ngưỡng 50% / 80% profit — bao nhiêu SKU?

Cách đếm: sắp xếp SKU theo `profit_pos = max(profit, 0)`, tính `cum_profit_pct`; đếm số SKU có `cum_profit_pct` **vẫn &lt; 50%** (hoặc &lt; 80%) — cùng logic `eda.py` / `build_sku_artifacts.py`.

| Cách tính profit | ~50% profit | ~80% profit | File artifact (`main/output/`) |
| :--- | ---: | ---: | :--- |
| **EDA** (`eda.py`: chỉ bỏ dấu `,` ở `Cost Amount`) | **99 SKU** | **653 SKU** | `top50pct_skus_eda.csv`, `top80pct_skus_eda.csv` |
| **Pipeline** (parse VN đầy đủ — **khớp WRMSSE / notebook**) | **213 SKU** | **1.403 SKU** | `top50pct_skus.csv`, `top80pct_skus.csv` |

**Nên dùng khi modeling:** **213** và **1.403** (pipeline). Số 99 / 653 chỉ đúng với EDA sơ bộ, không trùng weight metric.

*Tham khảo thêm (pipeline, cố định rank): top **99** SKU đứng đầu ≈ **39,7%** profit; top **653** ≈ **67,8%** — không đạt ngưỡng 50%/80%.*

# Metric: WRMSSE

Weighted Root Mean Squared Scaled Error – bình phương sai số dự báo được scale theo naive baseline, sau đó weighted trung bình theo profit của từng SKU. SKU lợi nhuận âm có weight = 0. → Đại khái nghĩa là: nếu dự báo sai khác nhiều so với thực tế thì bị trừ đi nhiều, nếu dự báo gần với thực tế thì bị trừ ít.

- RMSSE = 1: Ngang bằng naive baseline (dự đoán = ngày hôm qua)
- RMSSE < 1: Tốt hơn naive – mục tiêu đạt được
- RMSSE > 1: Tệ hơn naive – cần cải thiện

Tập trung tối ưu dự báo cho **~1.403 SKU** (= 80% profit, pipeline) sẽ có tác động lớn nhất đến WRMSSE. Các SKU còn lại ít ảnh hưởng điểm số.

Ưu tiên: làm tốt **~213 SKU** (~50% profit) trước, sau đó mở rộng tới ~1.403 SKU (~80%), rồi polish phần còn lại.

# Các vấn đề chính:
### 1. Sparsity cực cao: 
- Dữ liệu gốc là từng dòng giao dịch, không phải “mỗi ngày mỗi SKU một số”. Sau khi gộp theo ngày, nhiều SKU hầu như không bán: trung vị chỉ 6 ngày có bán trong 1.411 ngày (~0,4% ngày có giao dịch).

→ Ảnh hưởng:
- Model học pattern theo thời gian (lag, trend, mùa) rất khó vì chuỗi toàn 0 xen vài điểm bán. (Tất nhiên vẫn có thể xử lý bằng cách clean dữ liệu = 0, nhưng data sẽ ít nhiều bị ảnh hưởng)
- Dễ dự báo quá cao (mean bị lệch vì vài ngày bán lớn) hoặc quá thấp (toàn 0). (Ảnh hưởng trực tiếp do data long-tail: ~1.403 SKU chiếm 80% profit theo pipeline)
- Naive / LightGBM đều kém trên SKU sparse (vài ngày mới bán 1 lần).

**Cách xử lý đề xuất**: Gộp daily_qty, fill ngày không bán = 0; SKU sparse dùng mean vài tuần cuối hoặc Croston, không ép model phức tạp.
**Lưu ý**: “Fill ngày không bán = 0”: chuẩn cho demand forecasting; cần calendar đầy đủ mỗi SKU (1411 ngày hoặc range chung) trước khi tính RMSSE denominator (naive trên chuỗi liên tục). Croston / mean vài tuần cho SKU sparse: ổn cho SKU ít ảnh hưởng điểm; ~1.403 SKU top profit nên model mạnh hơn (LightGBM + lag/rolling trên daily).

### 2. Long-tail profit:
- Lợi nhuận tập trung ít SKU (pipeline): **~213 SKU ≈ 50% profit**, **~1.403 SKU ≈ 80% profit**. Phần còn lại ~14,5k SKU chia ~20% profit. *(EDA nhanh ghi 99 / 653 — khác cách parse số, xem bảng trên.)*

→ Ảnh hưởng:
- Metric WRMSSE = trung bình RMSSE có trọng số theo profit → sai dự báo ở SKU lãi lớn làm điểm tổng tệ hơn nhiều so với sai ở SKU nhỏ. -> Model phải đặc biệt chú ý vài SKUs profit cao, tránh overpredict sparse SKUs.
- Tối ưu “trung bình cho mọi SKU” không tối ưu điểm thi. -> Tập trung cho ~1.403 SKU profit cao (pipeline).
- Nên ưu tiên thời gian: ~213 SKU (~50%) → ~1.403 SKU (~80%) → polish phần còn lại.

**Cách xử lý đề xuất**: List top profit từ EDA, model tốt hơn (LightGBM / tune mean window) cho nhóm này, còn SKU nhỏ dùng rule đơn giản.

### 3. Return transactions:
- Khoảng 37k dòng <=> 5.3% có quantity < 0 là khách trả hàng

→ Ảnh hưởng:
- Nếu không gộp/khấu trừ (net) các giao dịch bán và trả hàng theo từng ngày, thì có thể một ngày có rất nhiều dòng dữ liệu ghi nhận cả bán và trả. Khi đó, model sẽ dễ bị hiểu nhầm là trong ngày này nhu cầu của khách hàng là âm (tức là khách trả nhiều hơn mua), mặc dù thực chất chỉ là do dữ liệu chưa được tổng hợp lại đúng cách.
- Nếu bạn chỉ cộng trực tiếp các giao dịch (bán và trả) mà không tổng hợp (aggregate) lại theo từng ngày, thì khi tính các feature như lag hoặc rolling (dịch chuyển hoặc trung bình trượt), kết quả sẽ bị sai lệch do nhiều giao dịch trong cùng một ngày không được gộp lại thành tổng số lượng bán ròng cho ngày đó. Nói cách khác là cần phải tổng hợp quantity theo ngày trước khi tính các đặc trưng chuỗi thời gian, nếu không các feature sẽ không phản ánh đúng thực tế.
- Dự báo cuối phải ≥ 0: vì trong thực tế số lượng bán ròng (gồm cả trả hàng/returns) không thể nhỏ hơn 0. Returns sẽ khiến doanh số thực tế (net sales) nhỏ hơn tổng doanh số bán ra (gross sales), nhưng tổng lại vẫn tối thiểu là 0, không thể dự báo thành số âm.

**Cách xử lý đề xuất**: đặt biến `daily_qty` = sum(Quantity) theo (Date, ItemCode) — trả hàng tự trừ trong cùng ngày. Khi dự báo vẫn clip ≥ 0.


## Lưu ý:
Hai dòng submission / SKU: mỗi dòng 28 cột F1..F28 nhưng khung thời gian khác (public 2025-09-06→10-03, private 2025-10-04→10-31). Note nói “56 ngày” đúng nhưng nên ghi rõ: hai forecast 28 ngày riêng, không phải một dòng 56 cột.

# Các mục tiêu chính:

## Mục tiêu bài toán

| Mục tiêu | Tiêu chí |
| :--- | :--- |
| Dự báo demand | `daily_qty` cho **15,972 SKU** × **56 ngày** (2 window × 28 ngày) |
| Metric | **WRMSSE càng thấp càng tốt**; mục tiêu từng SKU: **RMSSE < 1** (tốt hơn naive) |
| Ưu tiên điểm | Top **~213 SKU** (~50%) → **~1.403 SKU** (~80%) trước (pipeline / WRMSSE), rồi polish phần còn lại |
| Submission hợp lệ | Đủ 31,944 `id`, giá trị **float ≥ 0**, không trùng `id`, khớp `sample_submission.csv` |

## Mục tiêu theo giai đoạn

1. **EDA & hiểu data** — Hoàn thành phân tích sparsity, profit weight, returns (phần trên). -> ok rồi
2. **Baseline có điểm** — Pipeline end-to-end + file submission đầu tiên (notebook `auto_parts_demand_eda_wrmsse_baseline.ipynb`).
3. **Cải thiện top SKU** — Model/features tốt cho ~1.403 SKU (~80% profit, pipeline).
4. **Tối ưu toàn bộ** — Xử lý SKU sparse, tune, validate local trước khi nộp Private.

