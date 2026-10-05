# Chặng 4 — Ablation công thức huấn luyện trên Kaggle

Chặng 3 đã đối chiếu CRC, code nguyên byte, toàn bộ val predictions và logits bằng evaluator gốc.
Cả năm backbone hoàn tất 10 epoch; ConvNeXt-Tiny dẫn đầu macro-F1 val 0.964603, top-1
0.972579, FP32 batch-1 p50 11.35 ms trên T4. DeiT-Small đạt macro-F1 0.958794.
Chọn ConvNeXt-Tiny cho ablation theo macro-F1 val; một seed chưa xác lập ý nghĩa thống kê.
Kết quả đã lưu ở `validation/stage3_kaggle/`, gồm `verification.json`; ZIP nhỏ không có checkpoint.

## Chạy

1. Import `code/kaggle_stage4.ipynb`, gắn dataset ảnh hiện có, bật GPU và Internet.
2. Giữ IMAGES_DIR hiện tại nếu đúng. Không cần dataset code hoặc output chặng 3.
3. Save Version → Save & Run All. Các test và toàn bộ train chạy trong notebook Kaggle.
4. Tải `/kaggle/working/stage4_results.zip` về gốc source code và giữ output có checkpoint
   `deepweeds_stage4/outputs/runs`. ZIP nhỏ không chứa best.pt/last.pt.

Notebook tự chứa code, test, CSV nguyên gốc, biên nhận EDA và bằng chứng chọn backbone chặng 3.
Không đọc lại 17.509 ảnh để chạy EDA nếu biên nhận/path/CSV vẫn khớp. Không mở test.

## Kế hoạch

| Run | Trục | Khác T00 |
|---|---|---|
| T00 | nền | Pretrained, fine-tune toàn bộ, crop + flip, CE |
| T01_scratch | A: khởi tạo | Khởi tạo ngẫu nhiên toàn bộ |
| T02_frozen | A: khởi tạo | Pretrained backbone đóng băng, chỉ train head |
| T03_randaug | B: augmentation | Thêm torchvision RandAugment mặc định (2 ops, magnitude 9) |
| T04_cutmix | B: augmentation | Thêm CutMix mỗi batch, alpha=1, nhãn trộn theo diện tích thực |
| T05_smoothing | C: loss | CE label smoothing epsilon=0.1 |
| T06_focal | C: loss | Focal gamma=2, không trọng số lớp |
| T07_combined | kết hợp | Sau đủ bảy run, chọn giá trị tốt nhất từng trục nếu hơn T00 trên val |

Giữ cố định fold0, seed0, ConvNeXt-Tiny `convnext_tiny.fb_in1k`, ảnh224, batch32,
10 epoch, head Normal(0,0.01)/bias0, AdamW LR backbone1e-4/head1e-3, weight decay0.05,
warmup1 epoch + cosine, AMP, không EMA/sampler. Mỗi run bắt đầu từ pretrained gốc
(hoặc scratch theo trục A), không tiếp tục checkpoint từ run khác. Chọn epoch theo macro-F1 val,
hòa lấy epoch sớm hơn. Val FP32 resize256 + center crop224, mean/std pretrained.

T00 chạy mới theo cùng recipe B02 chặng 3 để có checkpoint ngay trong output chặng 4 và
so sánh các ablation trong cùng phiên. Delta luôn dùng T00 chặng 4; không giả định kết quả
lặp lại bằng tuyệt đối qua môi trường GPU. Không cần tải checkpoint chặng 3 làm input.
Scratch giữ cùng LR/10 epoch để đo một yếu tố; kết quả chỉ phản ánh ngân sách này,
không kết luận scratch đã hội tụ hoặc không thể đạt tốt hơn khi tối ưu riêng.

Kết hợp dùng nền T00 cố định cho mọi trục, không đổi nền theo thứ tự. Hòa hoặc kém hơn giữ
baseline; lưu quyết định vào combination_selection.json trước khi train. Chênh lệch rất nhỏ
chỉ là ứng viên cho kiểm tra nhiều seed sau này. Nếu mọi trục giữ nền, T07 là lần lặp nền
và được ghi rõ, không gọi là kết hợp có cải thiện. Cấu hình kết hợp có thể kém hơn từng yếu tố.
Leader là run hoàn tất tốt nhất trong cả tám, không mặc định chọn T07 cho chặng 5.

## Output và tiếp tục

- training.csv, training_ranked.csv: metric val, delta so với T00, F1/recall cả9 lớp, trạng thái.
- training_ablation.png, curves/Txx_seed0.png và history từng run.
- runs/<exp_id>/seed0/: best.pt, last.pt, config, môi trường, logits, summary.
- predictions/<exp_id>_seed0_val.csv theo eval.py gốc; không có test predictions.
- combination_selection.json, combined_config.json, suite_identity.json và stage4_summary.json.
- stage4_results.zip: kết quả nhỏ, log, code và manifest sau mỗi run, kể cả khi run sau lỗi.

Ngân sách mềm mặc định6 giờ; theo thời gian B02 chặng3, tám run khoảng1.6 giờ train+val
nếu tốc độ tương tự. Đây chỉ là ước lượng, chưa gồm setup/tải model/lưu output; scratch,
backbone frozen và augmentation có thời gian khác. Code dùng cuda:0, một T4.

Nếu chưa hoàn tất, add output phiên chặng4 làm input, đặt PREVIOUS_OUTPUT tới
`deepweeds_stage4/outputs` từ phiên đó rồi chạy lại đúng notebook. Không dùng output chặng3.
Phải giữ checkpoint và suite_identity; ZIP nhỏ không đủ resume. Đặt PREVIOUS_OUTPUT=None
khi chạy lại cell trong cùng session. Ngân sách được kiểm tra ở ranh giới epoch, không
ngăn được ngắt cứng Kaggle. Nếu OOM/lỗi, gửi traceback và ZIP từng phần; không tự đổi recipe.

## Kiểm tra

Các cell nhúng được kiểm tra cú pháp và checksum tại local; không train hoặc chạy test
pipeline local trong chặng4. Notebook chạy test_pipeline.py, test_stage3.py và test_stage4.py
trên Kaggle trước train; log được đưa vào ZIP. Chưa có kết quả huấn luyện chặng4.
Chặng6 mới chạy baseline/final ít nhất3 seed và đánh giá test sau khi chốt trên val.

## Kết quả Kaggle đã xác nhận ngày 2026-10-05

Đủ8 run, mỗi run10 epoch trên toàn bộ train/val. Notebook không có error;
log test_pipeline, test_stage3 và test_stage4 đều PASS. Đã kiểm tra CRC, code/evaluator
nguyên byte, val filenames/nhãn, softmax từ logits, metric và delta/F1/recall cả9 lớp.
Archive và verification lưu ở `validation/stage4_kaggle/`. Test chưa đánh giá.
ZIP nhỏ không chứa checkpoint, nên chưa xác nhận trọng số best.pt từ archive này.

| Run | Macro-F1 val | Top-1 val | ECE | Best epoch |
|---|---:|---:|---:|---:|
| T00 | 0.964603 | 0.972579 | 0.007417 | 9 |
| T01_scratch | 0.332554 | 0.589831 | 0.024400 | 8 |
| T02_frozen | 0.716261 | 0.783776 | 0.051144 | 10 |
| T03_randaug | 0.956552 | 0.966295 | 0.007782 | 8 |
| T04_cutmix | 0.959314 | 0.969437 | 0.016896 | 9 |
| T05_smoothing | 0.964664 | 0.974007 | 0.094831 | 10 |
| T06_focal | 0.960010 | 0.968580 | 0.060660 | 10 |
| T07_combined | 0.964664 | 0.974007 | 0.094831 | 10 |

T05 là ứng viên dẫn đầu theo tiêu chí macro-F1 đã định; delta so với T00 chỉ0.0000612
(0.00612 điểm phần trăm), chưa đủ kết luận cải thiện ổn định từ một seed.
ECE và NLL của T05 cao hơn T00; cần so sánh calibration khi làm chặng5.
Giữ T00 làm mốc và T05 làm ứng viên, chưa đóng băng cấu hình cuối.

Lựa chọn từng trục giữ nền cho A/B và chọn smoothing cho C. Do đó T07 có cùng recipe,
seed và kết quả với T05; đây là lần lặp một thay đổi, không phải bằng chứng hiệu ứng
cộng dồn nhiều yếu tố hoặc seed độc lập. Nếu cần kiểm tra tương tác hai yếu tố để bổ sung
báo cáo, phải khai báo một run riêng (ví dụ CutMix+smoothing); chưa có kết quả run đó.

Chặng5 cần output notebook chặng4 có:
`deepweeds_stage4/outputs/runs/T05_smoothing/seed0/best.pt` và
`deepweeds_stage4/outputs/runs/T00/seed0/best.pt`.
Giữ output này trên Kaggle và add làm input notebook tiếp theo; ZIP nhỏ chỉ đủ kiểm tra
kết quả, không đủ suy luận bằng model đã train.
