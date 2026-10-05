# Chặng 3 — So sánh backbone

## Thao tác Kaggle

1. Import bản mới `code/kaggle_stage3.ipynb`; chỉ gắn dataset ảnh `data-labd2` hiện có.
2. Bật GPU + Internet. Dùng một T4 qua `cuda:0`; hai T4 không tự cộng VRAM.
3. Giữ `IMAGES_DIR=/kaggle/input/datasets/luuquangkhai/data-labd2/images` nếu đường dẫn không đổi.
4. Chọn `SESSION_BUDGET_HOURS` theo hạn mức phiên/tài khoản. Mặc định 6 giờ là ngân sách
   mềm do notebook tự đặt, không phải cam kết về Kaggle. Ngân sách này bắt đầu tính ở vòng train;
   thời gian khởi tạo và test trước đó là phần cộng thêm.
5. Dùng **Save Version → Save & Run All**. Không cần upload ZIP code; không chạy lại EDA/overfit.
6. Khi xong, tải `stage3_results.zip` về gốc repo. Giữ output phiên có checkpoint.
   Nếu lỗi, gửi traceback và ZIP kết quả từng phần; không đổi riêng batch/LR/epoch để vượt lỗi.

## Kế hoạch cố định

| ID | Backbone và tag pretrained |
|---|---|
| B01_resnet50 | resnet50.a1_in1k |
| B02_convnext_tiny | convnext_tiny.fb_in1k |
| B03_deit_small | deit_small_patch16_224.fb_in1k |
| B04_efficientnet_b0 | efficientnet_b0.ra_in1k |
| B05_mobilenetv3 | mobilenetv3_large_100.ra_in1k |

Tất cả dùng fold 0, seed 0, 10 epoch, batch 32, ảnh 224×224, pretrained fine-tune toàn bộ,
head Normal(0, 0.01)/bias 0; CE, AdamW, LR backbone 1e-4/head 1e-3, weight decay 0.05
(loại norm/bias), warmup 1 epoch + cosine theo bước, AMP, không EMA/Mixup/CutMix/sampler.
Train: RandomResizedCrop bicubic + horizontal flip. Val: resize cạnh ngắn 256, center crop 224,
FP32; mean/std lấy từ pretrained config và được ghi vào output. CSV split giữ nguyên nhãn.
Mỗi model chọn checkpoint bằng macro-F1 val cao nhất; hòa lấy epoch sớm hơn.

Các trọng số pretrained được tạo bởi recipe khác nhau; ghi tag để không nhầm rằng đây là
so sánh kiến trúc thuần túy. Không dùng checkpoint một epoch chặng 2 làm khởi tạo cho B05:
cả năm backbone đều bắt đầu từ pretrained gốc và head mới.

## Output

- `backbones.csv`: đủ năm dòng, trạng thái pending/paused/completed và số liệu có thật.
- `backbones_ranked.csv`: chỉ run hoàn tất đủ 10 epoch và profiling; xếp theo macro-F1 val,
  hòa dùng p50 latency. Khi chưa đủ năm model, bảng chỉ là kết quả tạm thời.
- `backbone_tradeoff.png`: macro-F1 val theo độ trễ; không có kết quả test.
- `stage3_summary.json`: complete chỉ khi đủ năm run hoàn tất và profiling.
- `runs/<exp_id>/seed0/`: config, environment, preprocessing, history, best/last checkpoint,
  val_logits, curves.png, summary, profile.json. Checkpoint giữ trong output, không gói vào ZIP nhỏ.
- `predictions/*_val.csv`: toàn bộ val theo evaluator gốc, dùng kiểm tra lại số liệu.
- `stage3_results.zip`: bảng, log, đồ thị, predictions, code, manifest; cập nhật sau mỗi model.

Thời gian trong bảng là **train + val mỗi epoch**, chưa gồm tải pretrained, lưu checkpoint,
profiling và xuất output. Chặng 2 đo MobileNetV3 khoảng 83.85 giây/epoch, không dùng con số
này suy thời gian cho các backbone khác. Một seed chỉ đủ sàng lọc, chưa chứng minh chênh lệch nhỏ.

## Profiling và latency

GMAC đếm Conv2d, Linear và hai tích attention QK^T/AV của DeiT. Một MAC là một phép nhân-cộng;
không nhân đôi thành FLOPs. Bỏ qua bias, activation, norm, pooling, softmax và phép toán từng phần tử.
Counter hỗ trợ năm kiến trúc trong danh sách, có test tích attention độc lập; không tự áp dụng sang
mọi model timm khác. Đây là ước lượng MAC nhất quán, không thay thế đo tốc độ.

Độ trễ đo forward-only với input đã nằm trên GPU, FP32 batch 1, model.eval(), inference_mode,
10 lượt warmup, 100 lượt đo, synchronize trước/sau. Báo p50/p95/p99 và ghi GPU, torch, dtype.
Không tính đọc ảnh, transform, copy CPU→GPU. Đây là phép đo sơ bộ cho bảng backbone;
chặng 5 sẽ so sánh đầy đủ kỹ thuật suy luận, batch lớn hơn và các dtype.

## Resume hoặc lỗi

Không đổi notebook/code/recipe khi resume. Chọn output của phiên **chặng 3** đã lưu làm input,
đặt `PREVIOUS_OUTPUT` tới thư mục `deepweeds_stage3/outputs` của phiên đó. Không trỏ vào output
chặng 2. Cell đầu chép output; sau đó đặt lại `PREVIOUS_OUTPUT=None` nếu chạy lại cell trong session.
Checkpoint của run hoàn tất được bỏ qua; run dở tiếp tục ở ranh giới epoch.

Phải giữ `suite_identity.json` và các checkpoint. `stage3_results.zip` không chứa trọng số,
nên chỉ có ZIP nhỏ là chưa đủ resume. Không cần nén lại toàn bộ checkpoint sau mỗi model;
giữ chúng trong output đã lưu của notebook hoặc tải riêng về.

Ngân sách thời gian là cơ chế hợp tác ở ranh giới epoch, có dự phòng cho xuất output, không ngăn
được Kaggle ngắt cứng. Epoch đầu chưa có ước lượng thời gian. Nếu OOM hoặc tải pretrained lỗi,
vòng chạy dừng và xuất các kết quả đã có; gửi lỗi để xử lý. Không tự giảm batch cho riêng một model.

## Trạng thái

Đã xác nhận kết quả Kaggle ngày 2026-10-05: đủ năm backbone, mỗi run 10 epoch.
CRC archive, code, evaluator nguyên byte, toàn bộ val filenames/nhãn và xác suất từ logits
đã đối chiếu; không có test predictions. Kết quả ở `validation/stage3_kaggle/`.
ZIP nhỏ không chứa checkpoint, nên không xác nhận trọng số từ archive này.

| Backbone | Macro-F1 val | Top-1 val | p50 FP32 batch1 (ms) |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 0.964603 | 0.972579 | 11.352 |
| DeiT-Small | 0.958794 | 0.969437 | 12.547 |
| EfficientNet-B0 | 0.919067 | 0.939160 | 8.643 |
| MobileNetV3 | 0.887299 | 0.911740 | 7.110 |
| ResNet50 | 0.844266 | 0.883462 | 6.239 |

Chọn ConvNeXt-Tiny cho ablation [chặng 4](STAGE4.md) theo macro-F1 val.
Đây là sàng lọc seed0; khác biệt nhỏ chưa được kiểm chứng nhiều seed.
