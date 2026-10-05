# Chặng 2 — kiểm tra pipeline train trên Kaggle

## Bạn cần làm

1. Giữ dataset ảnh đang dùng, không upload lại `images.zip`.
2. Dùng bản mới nhất của `code/kaggle_stage2.ipynb`: notebook đã tự chứa toàn bộ code,
   test, CSV nhãn và biên nhận chặng 1. Không cần upload ZIP hoặc tạo dataset code.
3. Import notebook này lên Kaggle và chỉ gắn dataset ảnh hiện tại.
4. Bật GPU và Internet. Notebook dùng `cuda:0`, tức một T4; hai T4 không tự gộp VRAM.
5. Cell đầu dùng đường dẫn ảnh đã xác nhận:
   `/kaggle/input/datasets/luuquangkhai/data-labd2/images`.
   Nếu Kaggle hiển thị đường dẫn khác, sửa biến `IMAGES_DIR`.
   Các cell tiếp theo tự ghi module `.py` và CSV vào working, kiểm tra SHA256 nguyên byte.
6. Dùng **Save Version → Save & Run All** để chạy cả notebook. Lượt này chỉ kiểm tra
   pipeline; chưa chạy toàn bộ bài lab hay đánh giá test.
7. Khi hoàn tất, tải `stage2_results.zip` từ output về gốc repo và báo cho tôi.
   Giữ `stage2_checkpoints.zip` làm bản dự phòng; chưa cần gửi file lớn này.
   Nếu cell lỗi, gửi traceback và các output đã có; không bỏ qua lỗi để chạy bước sau.

Notebook mới không quét đệ quy dataset ảnh, không giải mã lại toàn bộ ảnh để EDA.
Khi đường dẫn và checksum CSV khớp biên nhận chặng 1, chỉ kiểm tra split từ CSV.
Ảnh được đọc theo từng batch. Giả định dataset ảnh không đổi phiên bản/nội dung sau chặng 1;
nếu ảnh bị thay đổi, cần chạy lại chặng 1 và cập nhật biên nhận.
Điều này giảm công việc trong notebook, không đảm bảo giảm thời gian Kaggle khởi tạo tài nguyên.

## Notebook sẽ làm gì?

| Bước | Kết quả cần đạt |
|---|---|
| Môi trường | GPU nhận được, timm 1.0.30, ghi pip-freeze |
| Test CPU | Các kiểm tra loss, CutMix, frozen BN, EMA, split, resume đều đạt |
| Overfit | 9 ảnh train, mỗi lớp 1 ảnh, CE < 0.1 và accuracy = 1 trong tối đa 200 bước |
| S02_resume | 18 ảnh train + 9 ảnh val, 64×64, scratch; dừng sau epoch 1 rồi resume đến epoch 2 |
| S01_full_epoch | Train đủ 10.501 ảnh (theo loader), val đủ 3.501 ảnh, MobileNetV3 pretrained, 224×224, 1 epoch |
| Xuất output | Log, config, checkpoint, val logits/dự đoán và biểu đồ; không có dự đoán test |

Overfit có LR 1e-3 và weight decay 0 để chẩn đoán. S01 dùng warmup 0.1 epoch vì chỉ có
một epoch. Những lần chạy này không phải số liệu so sánh backbone. Công thức chặng 3
sẽ dùng exp_id mới, 10 epoch, warmup 1 epoch và cùng thiết lập cho mọi backbone.
Không dùng mức accuracy của một epoch để kết luận mô hình tốt/xấu.

## Code đã triển khai

- `dataset.py`: CSV/split, transform, DataLoader, seed worker; giữ nhãn split gốc.
- `model.py`: timm, freeze, head LR, loại weight decay cho norm/bias (kể cả head bias).
  Vì head bias cần weight decay 0, có thể có bốn nhóm thay vì ba nhóm gợi ý ở starter.
  Head phân loại mới khởi tạo Normal(0, 0.01), bias 0, dùng cùng quy tắc cho các backbone.
  Đã kiểm tra đầu vào chuẩn hóa đúng; head mặc định của timm MobileNet với 9 lớp có độ lệch
  chuẩn lớn (~0.19) và cho CE ban đầu ~5.8 trên batch kiểm tra. Head nhỏ giúp tránh
  khác biệt thang logit do kiến trúc. Đây là lựa chọn recipe được ghi trong Config.
- `losses.py`: CE, label smoothing, focal, class weights, Mixup/CutMix.
- `train.py`: AMP, AdamW, warmup/cosine theo bước, EMA, chọn best macro-F1 val
  (hòa lấy epoch sớm hơn), log và checkpoint.
- `smoke_checks.py`: xem augmentation, kiểm tra loss và overfit một batch train.
- `tests/test_pipeline.py`: test phần triển khai, độc lập với test bộ khung ở gốc repo.

`eval.py` giữ nguyên. `inference.py`/`benchmark.py` vẫn là bộ khung cho chặng 5.
Đánh giá val hiện dùng FP32 và CE thuần để so sánh nhất quán; loss train là objective đã chọn.
GMAC chưa đo trong chặng 2. Hàm profiling có sẵn là ước lượng bằng thop, cần xác minh
khả năng đếm attention trước khi so sánh transformer ở chặng 3.

## Lưu và chạy tiếp

Mỗi run nằm ở `outputs/runs/<exp_id>/seed<k>/`:

- `last.pt`: model, optimizer, scheduler, AMP scaler, EMA nếu có, RNG, generator của
  DataLoader, lịch sử, best weights và dấu vân tay cấu hình/code/CSV.
- `best.pt`: trọng số suy luận tốt nhất và thông tin tiền xử lý.
- `history.csv`, `config.json`, `environment.json`, `preprocessing.json`, `split_check.json`.
- `val_logits.npz`, `curves.png`, `summary.json`.
- Dự đoán val ở `outputs/predictions/`.

Checkpoint được thay file nguyên tử sau mỗi epoch. Nếu bị ngắt giữa epoch, resume chạy lại
epoch đó từ checkpoint epoch trước. Chỉ load checkpoint do workflow này tạo.
Giữ nguyên tổng `epochs`, seed, recipe và code khi resume. Đổi cấu hình/code/CSV phải dùng
exp_id mới; notebook từ chối trộn checkpoint không khớp. Phép thử CPU đã kiểm tra resume
khớp chạy liên tục, nhưng không cam kết kết quả bit-for-bit giữa phần cứng/phiên bản khác nhau.

Trong cùng session, chạy lại cell `run(...)` để resume; run hoàn tất sẽ được bỏ qua.
Ở session mới, gắn **output phiên notebook đã lưu** làm input, đặt `PREVIOUS_OUTPUT` tới
thư mục `outputs` của phiên cũ ở cell đầu. Cell này chép output sang working trước khi resume.
Sau lần chép đầu, đặt lại `PREVIOUS_OUTPUT=None` nếu chạy lại cell để tránh ghi đè checkpoint mới.
Chỉ lưu file trong `/kaggle/working` chưa phải bản sao lưu ngoài phiên: cần lưu phiên/output
và kiểm tra file đã xuất, hoặc tải về trước khi kết thúc session.

`max_run_seconds` là ngân sách mềm: dự đoán thời gian epoch tiếp theo từ các epoch đã đo
và dừng ở ranh giới epoch. Không bảo đảm tránh giới hạn cứng của Kaggle, đặc biệt epoch đầu.
Notebook chặng 2 chưa khởi chạy các vòng thí nghiệm dài.

## Kiểm tra local

Từ gốc repo, trên Windows:

```powershell
$env:PYTHONUTF8='1'
python submissions/02599_LuuQuangKhai/tests/test_pipeline.py
python -m unittest discover -s tests
```

Test pipeline cần torch, torchvision, timm, numpy, pandas, pillow, matplotlib.
Phiên bản torch/torchvision của máy local có thể khác Kaggle; notebook kiểm tra lại trước khi train.
Thử nghiệm test resume dùng model nhỏ và ảnh tổng hợp, không phải kết quả DeepWeeds.

## Trạng thái

Code đã triển khai và kiểm tra CPU tại local ngày 2026-10-04:

- 9/9 test pipeline đạt, gồm resume khớp trọng số với chạy liên tục trên CPU.
- 38/38 test gốc của repo đạt, evaluator giữ nguyên từng byte.
- Tải được pretrained `mobilenetv3_large_100.ra_in1k`.
- Overfit 9 ảnh train thật với head Normal(0, 0.01): CE ban đầu 2.1645,
  CE cuối 0.001212, accuracy 100% sau 10 bước. Đây chỉ là kiểm tra một batch,
  không phải metric đánh giá DeepWeeds. Log nằm ở `validation/stage2_cpu_smoke/`.
- Notebook tự chứa code từ ngày 2026-10-05; các file nhúng được kiểm tra SHA256.
  File ZIP cũ chỉ còn là gói tùy chọn, notebook không phụ thuộc vào nó.

## Xác nhận Kaggle ngày 2026-10-05

Đã nhận `stage2_results.zip`; kiểm tra archive, SHA256 code theo manifest và đối chiếu
code với bản local. Kết quả gốc được lưu trong `validation/stage2_kaggle/`.
Đã tính lại metric từ CSV bằng evaluator gốc, đối chiếu xác suất với logits và đối chiếu
tên file/nhãn với val CSV. Chặng 2 hoàn thành; có thể chuyển sang chặng 3.

| Kiểm tra | Kết quả |
|---|---|
| Phần cứng | Một Tesla T4, CUDA 12.8, PyTorch 2.11.0+cu128, timm 1.0.30 |
| Test pipeline trong notebook | 9/9 đạt |
| Overfit 9 ảnh train | 10 bước; CE 0.001251; accuracy 100% |
| S02_resume | Hoàn tất 2 epoch, best epoch 1; 18 ảnh train, 9 ảnh val |
| S01_full_epoch | 10.501 ảnh train, 3.501 ảnh val; AMP bật; 329 bước optimizer |
| Macro-F1 val S01 | 0.6476083016 |
| Top-1 val S01 | 0.7523564696 |
| ECE val S01 | 0.04831637 |
| Thời gian epoch S01 (train + val) | 83.85 giây |
| Test | Không đánh giá; không có file dự đoán test |

S02 có một optimizer step bị AMP bỏ qua ở epoch đầu; S01 không có step bị bỏ qua.
Đây là kết quả chẩn đoán, không phải thí nghiệm backbone 10 epoch. Ước lượng tuyến tính
riêng MobileNetV3: 10 × 83.85 giây ≈ 14 phút cho train/val, chưa tính overhead xuất output
và tải trọng số. Không suy thời gian này sang backbone khác.

Archive kết quả không chứa checkpoint: đã xác nhận quá trình resume từ log và kết quả,
chưa trực tiếp nạp checkpoint GPU của phiên này. Giữ `stage2_checkpoints.zip` dự phòng.
