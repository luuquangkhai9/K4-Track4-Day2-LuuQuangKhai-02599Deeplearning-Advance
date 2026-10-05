# Chặng 7 — Hoàn tất bài nộp

Đã tổng hợp kết quả các chặng Kaggle, kiểm tra predictions và raw logits đã lưu,
đối chiếu evaluator gốc, xây results.xlsx và hoàn thiện report.md/README.md.
Không train lại, không fit trên test và không đánh giá thêm cấu hình trên test.

## Sản phẩm

- results.xlsx: bảy sheet Summary, Backbones, Training, Inference, Final, PerClass, Latency.
- report.md: phương pháp, so sánh, calibration, phân tích lỗi và giới hạn.
- curves/: 21 đường cong theo epoch và một overfit diagnostic, kèm manifest nguồn.
- code/: mã nguồn và notebook Kaggle, evaluator gốc, audit_submission.py.
- predictions/: kết quả validation/screening và ba seed test của final/baseline/uncalibrated.
- README.md: môi trường, link Kaggle, cách tái lập và kiểm tra từ CSV.

Notebook đã chạy: https://www.kaggle.com/code/luuquangkhai/day2-lab-phase2

## Kết quả và kiểm tra

Final test macro-F1 0.9694669 ± 0.0019568, top1 0.9764281 ± 0.0005936 (ba seed;
std mẫu ddof1). Baseline macro-F1 0.9586004 ± 0.0014840. Final p95 batch1 max
17.921ms trên Tesla T4, không tính tiền xử lý và truyền CPU→GPU.
Evaluator đề xuất 20/20 phần I; đây không phải điểm toàn bộ rubric.

Audit xác nhận sáu test receipts, một lượt forward/test/run, hashes raw logits,
calibrated/uncalibrated giữ argmax, đúng filenames/labels/splits, mean/std và
latency từ samples. Workbook đã recalculation, thử thay đổi một giá trị rồi khôi
phục, kiểm tra nguồn và render từng sheet; chưa kiểm tra trong Excel native.

T07 lặp lại một yếu tố loss, chưa thực hiện tương tác nhiều yếu tố. Giới hạn này
được ghi rõ trong bảng và báo cáo. Trọng số không có trong ZIP kết quả nhỏ;
cần giữ Kaggle saved output nếu muốn sử dụng checkpoint.

Gói ZIP nằm cạnh thư mục bài nộp, có prefix 02599_LuuQuangKhai/.
submission_manifest.json ghi dung lượng và SHA256 từng file được đóng gói,
ngoại trừ chính manifest. Có thể chạy python code/audit_submission.py để kiểm
tra lại mà không dùng GPU. Gói loại caches, local smoke fixtures và trọng số.
