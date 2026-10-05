# Bài làm DeepWeeds — 02599 Lưu Quang Khải

## Trạng thái

Chặng 1 đã hoàn thành tại local và Kaggle; output Kaggle đã đối chiếu ngày 2026-10-04.
Chặng 2 đã triển khai dataset/model/loss/train và notebook kiểm tra; xem [STAGE2.md](STAGE2.md).
Notebook chặng 2 tự chứa code và nhãn: chỉ import `.ipynb`, gắn dataset ảnh, không cần dataset code riêng.
Chặng 2 đã xác nhận trên Kaggle ngày 2026-10-05, kết quả ở `validation/stage2_kaggle/`.
[Chặng 3](STAGE3.md) đã hoàn tất trên Kaggle và đối chiếu ngày 2026-10-05: đủ năm backbone,
ConvNeXt-Tiny dẫn đầu macro-F1 val 0.964603. Kết quả ở `validation/stage3_kaggle/`.
[Chặng 4](STAGE4.md) đã hoàn tất tám run trên Kaggle, đối chiếu ngày 2026-10-05;
kết quả ở `validation/stage4_kaggle/`. T05 label smoothing dẫn đầu macro-F1 val0.964664,
chỉ hơn T00 0.0000612; ECE tăng lên0.094831. Giữ cả T05 và T00 để đối chiếu chặng5.
T07 lặp cùng recipe/seed T05, không phải seed độc lập hay kết hợp nhiều thay đổi.
[Chặng5](STAGE5.md) đã xác nhận16 cấu hình trên Kaggle. T05 +suy luận288 dẫn đầu:
macro-F1 val0.970583, top1 0.978006, p95 batch1 15.47ms. Evidence ở `validation/stage5_kaggle/`.
Đã chuẩn bị [chặng6](STAGE6.md): notebook `code/kaggle_stage6.ipynb` tự chứa code,
train lại F01(label smoothing, suy luận288+temperature) và T00(CE, suy luận224) với3 seed.
Chốt recipe trước test, một vòng test/run; std mẫu ddof1, chạy score/grade gốc.
Chưa có kết quả GPU/test chặng6.
Notebook Kaggle: sẽ điền link sau khi chạy. Không coi kiểm tra CPU local là kiểm tra GPU Kaggle.

## Chạy trên Kaggle

1. Tạo dataset **Private** chứa `images.zip` ở gốc repo.
2. Tạo dataset **Private** thứ hai chứa `kaggle_stage1_code.zip` ở gốc repo.
3. Tạo notebook bằng cách import `code/kaggle_stage1.ipynb` trong thư mục bài nộp này.
4. Add Input cả hai dataset. Notebook nhận cả ZIP nguyên và dữ liệu đã được Kaggle giải nén.
5. Bật Internet để cài thư viện nếu thiếu. Chặng 1 chạy được bằng CPU; bật GPU ở cell kiểm tra
   nếu muốn xác nhận sẵn môi trường cho chặng 2. Không cần giữ GPU chạy khi đang upload dữ liệu.
6. Run All. Nếu có nhiều nguồn code/ảnh giống nhau, bỏ nguồn cũ khỏi notebook rồi chạy lại.
7. Kiểm tra dòng `PASS`, bảng đếm và hai ảnh EDA. Lưu phiên notebook và tải
   `/kaggle/working/stage1_results.zip` về local.
8. Gửi `stage1_summary.json` hoặc toàn bộ `stage1_results.zip` để tiếp tục chặng 2.
   Nếu lỗi, gửi toàn bộ traceback và số cell; chưa chạy cell tiếp theo.

Không sửa CSV chia tập, không tự chia lại dữ liệu. Chỉ xem mẫu ảnh từ train.
Thư mục `/kaggle/input` là nguồn đọc; notebook chép code sang `/kaggle/working/deepweeds_stage1`.
Cell kiểm tra đọc đủ 17.509 ảnh nên có thể mất vài phút. Cell này không train.

## Đầu ra chặng 1

- `stage1_summary.json`: số ảnh từng tập, giao, hợp, checksum CSV, kích thước/kênh ảnh, môi trường.
- `class_counts.csv`: số lượng từng lớp theo split.
- `class_distribution.png`: biểu đồ phân bố lớp.
- `train_samples.png`: ba ảnh train mỗi lớp, lấy cố định với seed 0.
- `sample_manifest.csv`: nguồn của 27 ảnh minh họa.
- `label_discrepancies.csv`: khác biệt nhãn giữa CSV split và labels.csv, nếu có.
- `pip-freeze.txt`: thư viện trong phiên Kaggle.

`eda/local/` là kết quả kiểm tra thật tại local, không phải kết quả Kaggle.
`labels/` chứa bốn CSV gốc tải từ AlexOlsen/DeepWeeds; SHA256 được ghi trong summary.
`code/eval.py` là bản sao nguyên vẹn từ repo, không chỉnh sửa.

## Kết quả kiểm tra local ngày 2026-10-04

- Train: 10.501; val: 3.501; test: 3.507. Hợp: 17.509, mọi giao bằng 0.
- Đọc và giải mã thành công tất cả ảnh: RGB 256×256. MD5 ZIP đúng.
- Có **một khác biệt nhãn trong nguồn gốc**: `20170714-110407-3.jpg` có Label=0
  trong train_subset0.csv, nhưng Label=1 trong labels.csv. Không tự sửa CSV.
  Thống kê theo split có 1.126 Chinee apple và 1.063 Lantana; theo labels.csv là
  1.125 và 1.064. Đã lưu chi tiết để báo cáo/trao đổi với giảng viên.
- Kiểm tra tính toàn vẹn đạt PASS; PASS không có nghĩa nguồn không có khác biệt nhãn.
- 38 test sẵn có của repo đạt khi bật UTF-8 trên Windows (`$env:PYTHONUTF8='1'`).
  Các test này kiểm tra evaluator/bộ khung, chưa chứng minh pipeline train đã triển khai.
- Notebook đã kiểm tra cú pháp tại local; kết quả chạy Kaggle được xác nhận bên dưới.

## Xác nhận kết quả Kaggle ngày 2026-10-04

Đã nhận `stage1_results.zip`, kiểm tra CRC của archive và lưu kết quả ở `eda/kaggle/`.
Summary, số lượng lớp, danh sách 27 ảnh minh họa và khác biệt nhãn khớp kết quả local.
Bốn CSV nhãn trong archive khớp từng byte với bản local và SHA256 đã ghi.

- Trạng thái kiểm tra dữ liệu: PASS. Giữ nguyên khác biệt nhãn đã nêu ở trên.
- GPU: 2 × Tesla T4; Python 3.13.15; PyTorch 2.11.0+cu128; CUDA 12.8.
- NumPy 2.1.3; pandas 2.3.3; matplotlib 3.10.0. Chi tiết ở `eda/kaggle/pip-freeze.txt`.
- Nguồn ảnh: `/kaggle/input/datasets/luuquangkhai/data-labd2/images`.
- `zip_md5=null` là đúng vì Kaggle đọc thư mục ảnh đã giải nén; không tính checksum ZIP
  tại Kaggle. Kiểm tra decode đủ ảnh và kích thước/kênh đã đạt; không suy ra ảnh khớp
  từng byte với ZIP local từ các kiểm tra này.
- Sẵn sàng chặng 2. Nhìn thấy hai GPU không có nghĩa code tự dùng cả hai;
  kiểm tra pipeline ban đầu sẽ dùng một GPU trước.

## Chạy lại chặng 1 tại local

Từ gốc repo, cài numpy, pandas, matplotlib và pillow nếu thiếu rồi chạy:

```powershell
python submissions/02599_LuuQuangKhai/code/stage1.py --images images.zip --labels submissions/02599_LuuQuangKhai/labels --output submissions/02599_LuuQuangKhai/eda/local
```

## Các chặng tiếp theo

1. Chặng 2: pipeline train, kiểm tra một batch, một epoch và resume.
2. Chặng 3: năm backbone theo cùng recipe.
3. Chặng 4: ablation ba trục và một kết hợp.
4. Chặng 5: bốn phương pháp suy luận ngoài mốc, đo độ trễ.
5. Chặng 6: final và baseline với ít nhất ba seed; test sau khi chốt cấu hình trên val.
6. Chặng 7: xuất results.xlsx, report.md và đối chiếu bằng eval.py.
