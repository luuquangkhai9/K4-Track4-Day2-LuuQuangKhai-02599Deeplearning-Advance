# Chặng 6 — Final và baseline với ba seed, test một lần

## Cấu hình đã chốt từ val chặng5

- Backbone ConvNeXt-Tiny `convnext_tiny.fb_in1k`.
- F01: pretrained fine-tune toàn bộ, crop+flip, CE label smoothing0.1;
  train224, batch32,10 epoch, AdamW LR backbone1e-4/head1e-3, wd0.05,
  warmup1+cosine, AMP khi train. Suy luận FP32 một view288 (resize cạnh ngắn329).
- T00: cùng recipe train, CE không smoothing; suy luận FP32 một view224 (resize256).
- Cả hai chạy lại seed0,1,2 từ pretrained gốc, không warm-start từ chặng4.
- Mỗi checkpoint vẫn chọn bằng macro-F1 val224 trong quá trình train.
  Val cuối của F01 được suy luận288 để phù hợp cấu hình test; không chọn lại epoch theo test.
- F01 fit một nhiệt độ trên val288 riêng cho từng seed trước test; T00 giữT1.
  Không dùng nhiệt độ0.60926 của chặng5 ở224 cho288.

Chặng5 dẫn đầu T05+288: macro-F1 val0.970583, top1 0.978006, p95 batch1 15.47ms.
Temperature được chọn như quy tắc calibration bảo toàn argmax từ thí nghiệm I07;
fit lại trên288 chỉ dùng val, không dò thêm cấu hình accuracy. Val sau fit là metric trên
chính tập fit, chưa phải calibration holdout. Quy tắc không đổi theo kết quả test.
Evidence ở `validation/stage5_kaggle/verification.json`; quyết định ở `stage6_selection.json`.

## Chạy hoàn toàn trên Kaggle

1. Import `code/kaggle_stage6.ipynb`, chỉ gắn dataset ảnh hiện có, bật GPU+Internet.
2. Không cần gắn output chặng4/5 hoặc codeZIP: notebook tự chứa code, test, CSV và evidence.
3. Save Version → Save & Run All. Sáu run train mới chạy tuần tự trên cuda:0.
4. Sau train đủ6 run và lưu đủ6 val calibration/latency, notebook mới đánh giá test.
5. Tải `/kaggle/working/stage6_results.zip` về gốc sourcecode. Giữ output Kaggle có best.pt/last.pt.

Mặc định ngân sách mềm6h. Chặng4 khoảng75s/epoch gợi ý khoảng75 phút train+val224
cho6×10 epoch nếu tốc độ tương tự, cộng tải model, checkpoint, val288, test và export.
Đây chỉ là ước lượng, không phải bảo đảm thời gian/hạn mức Kaggle. Không giảm riêng
seed hoặc batch để vượt lỗi. Chỉ GPU0 được dùng dù phiên có hai T4.

## Một lượt test và resume

Trước train, frozen_recipe.json lưu chính xác recipe, seed, CSV/code hashes và phương pháp
suy luận. File đó được giữ trong ZIP. Resume yêu cầu cùng code/config/source dữ liệu.

Mỗi run ghi test_started.json trước khi mở ảnh test đầu tiên. Sau một vòng toàn bộ3507 ảnh,
logit được lưu nguyên tử cùng hash identity. F01 calibrated và F01_uncal cùng được xuất
**từ logit này**, không chạy model hai lần. Đã có raw cache thì resume chỉ xuất lại CSV/
metric, không đọc lại ảnh test hoặc chạy model. Không dùng test để đổi T, epoch hay recipe.

Nếu GPU/session ngắt giữa vòng test trước khi có cache đầy đủ, code dừng thay vì tự chạy
lại test. Gửi lỗi và các receipt để ghi nhận điều kiện bị ảnh hưởng; không xoá test_started.json.
Không mở kết quả test rồi quay lại tìm cấu hình mới. Đây là yêu cầu của GUIDE mục5.

Nếu dừng mềm trước khi xong, add output chặng6 làm input, đặt PREVIOUS_OUTPUT tới
`deepweeds_stage6/outputs`, dùng đúng notebook. Train resume từ last.pt ở ranh giới epoch;
run hoàn tất bỏ qua. Không dùng output chặng4/5 làm PREVIOUS_OUTPUT. ZIP nhỏ không chứa
checkpoint nên không đủ resume phần train. Giữ frozen_recipe, logits và receipt khi resume.

## Output

- predictions/T00_seed{k}_{val,test}.csv, F01_seed{k}_{val,test}.csv và
  F01_uncal_seed{k}_{val,test}.csv, k0,1,2, theo evaluator gốc.
- runs/...: config, môi trường, history, curves, best/last;
  final_evaluation/: val288/224 logits, T, metric, latency batch1/32, test logits/receipt.
- final_per_seed.csv, final_mean_std.csv, final_per_class.csv, final_per_class_mean_std.csv:
  scalar metric, F1/recall/precision từng lớp; std mẫu ddof1.
- confusions/: ma trận nhầm lẫn test riêng từng config/seed.
- official/: eval.py score cho T00/F01/F01_uncal, grade_I.json và log grade.
  Đầu vào latency grade là max p95 đo trên ba seed final, batch1 FP32 288+temperature,
 10warmup100lượt, synchronize trước/sau; decode/resize/normalize/CPU→GPU không tính.
- stage6_summary.json complete chỉ khi6 test receipt và score/grade/confusion hoàn tất.
- stage6_results.zip không chứa trọng số; cập nhật sau từng run và xuất phần đã có khi lỗi.

Kiểm tra local:8/8 test điều khiển/fixture đạt (không train thực), gồm khóa recipe,
khóa test sau lần bắt đầu, xuất uncal/cal cùng một forward, recovery từ cache, thứ tự
train→val→test, std từ prediction fixtures và lệnh score/grade chính thức trên fixture. Kaggle chạy lại pipeline, inference và
stage6 checks trước train. Kết quả GPU đã được nhận và kiểm tra tại chặng7;
results.xlsx/report tổng hợp từ kết quả thật, không chạy lại model trên test.

## Trạng thái hoàn tất

Đã nhận kết quả Kaggle, xác nhận đủ6 test receipts và3 seed mỗi cấu hình. Predictions/logits,
calibration/freeze, mean/std và grade gốc đã audit. Final test macro-F1 0.9694669±0.0019568,
top1 0.9764281±0.0005936, p95 batch1 max17.921ms; grade phầnI đề xuất20/20.
Bài nộp chặng7: results.xlsx, report.md, curves, predictions và README đã hoàn thiện.
