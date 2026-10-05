# Chặng 5 — Phương pháp suy luận trên Kaggle

Import `code/kaggle_stage5.ipynb`, gắn dataset ảnh và **output notebook chặng4 đã lưu**
làm input. Bật GPU+Internet; Save Version → Save & Run All. Không train lại model.

Notebook tự tìm outputs chặng4 trong /kaggle/input qua hai file best.pt của T00/T05.
Nếu có nhiều phiên phù hợp, điền STAGE4_OUTPUT tới thư mục `deepweeds_stage4/outputs`
của phiên muốn dùng. ZIP stage4_results chỉ có logit/kết quả, không đủ chạy model.
Checkpoint cần là `runs/T00/seed0/best.pt` và `runs/T05_smoothing/seed0/best.pt`.
Chỉ nạp checkpoint từ output của bạn; kiểm tra signature code/CSV/config trước suy luận.
I00 phải tái tạo logit val chặng4 trong sai số FP32 khai báo (atol1e-3, rtol1e-4).

| Phương pháp | View/độ phân giải | Cách gộp |
|---|---|---|
| I00 | Center crop224, resize cạnh ngắn256 | Softmax |
| I01_hflip | Crop224 + lật ngang, K2 | Trung bình xác suất |
| I02_5crop_prob | Resize cạnh ngắn256, 4 góc + giữa crop224, K5 | Trung bình xác suất |
| I03_5crop_logit | Cùng5 crop | Trung bình logit rồi softmax |
| I04_res256 | Resize cạnh ngắn293, center crop256 | Softmax |
| I04_res288 | Resize cạnh ngắn329, center crop288 | Softmax |
| I07_temperature | I00, fit một nhiệt độ trên toàn bộ val | Softmax(logit/T) |
| I08_amp | I00 + CUDA autocast FP16 | Softmax từ logit chuyển FP32 |

Chạy đủ8 phương pháp trên cả T00 CE và T05 label smoothing:16 dòng kết quả.
Không dùng test. T05 dẫn đầu rất ít ở chặng4 nhưng calibration kém hơn;
giữ T00 để thấy accuracy/calibration/latency tradeoff. Không mặc định chọn T05 cho final.

Temperature tối ưu NLL trên val với logT thuộc[-4,4], bao gồm ứng viênT1.
Kiểm tra argmax không đổi. ECE/NLL sau fit và đo trên chính val là **kết quả trên tập fit**,
không phải ước lượng calibration độc lập; phải nêu điều này trong báo cáo.
Chặng6 cần fit lại T trên val của từng seed rồi áp dụng sang test sau khi chốt recipe.
Không tối ưu trực tiếp ECE, nên không đảm bảo ECE luôn giảm khi NLL giảm.

Latency: CUDA synchronize trước/sau,10 warmup,100 lượt đo, batch1 và32,
p50/p95/p99 và thông lượng. Đo view/crop/flip, forward, gộp và temperature trên GPU;
không tính decode/resize/normalize/host-to-device. Input đã trên GPU; baseline cũng
bao gồm softmax, nên không so trực tiếp tuyệt đối với forward-only chặng3.
Crop view lấy từ tensor256 đã chuẩn hóa; giữ5 view chạy tuần tự, batch32 không nhân
lên160 trong một forward. Không suy độ trễ TTA bằng nhânK. ConvNeXt dùng LayerNorm,
không có thí nghiệm fuse BN; bn_fused=False. Không có EMA vì chưa trainEMA.

Checkpoint được đọc trực tiếp từ input, không copy vào output hoặc ZIP chặng5.
Giữ output chặng4 cho chặng6. Output chặng5 gồm inference.csv/ranked, đồ thị tradeoff,
report từng phương pháp, raw logit mỗi view, val predictionCSV, temperature và latency rawsamples.
Tải `stage5_results.zip` về sourcecode; không có trọng số hoặc predictions test.

Resume: add output chặng5 và đặt PREVIOUS_OUTPUT tới `deepweeds_stage5/outputs`, đồng thời
vẫn gắn output chặng4. Giữ đúng code/checkpoint; đã hoàn tất từng phương pháp được bỏ qua.
Nếu lỗi, ZIP phần đã xong được xuất trong finally, gửi kèm traceback. Chưa có GPU results chặng5.
Các kiểm tra pipeline và suy luận chạy ở Kaggle trước khi dùng checkpoint. Đã đạt6/6 unit test
suy luận trên CPU local (crop, gộp logit/prob, temperature, forward mô hình nhỏ, partial status,
fusion trên Sequential); không train local. Bundle được kiểm tra cú pháp và nguyên byte. Cấu hình cuối và kiểm tra>=3 seed thuộc chặng6.

## Kết quả Kaggle đã xác nhận ngày2026-10-05

Đủ16 cấu hình trên val, notebook không có error. Đã đối chiếu nguyên byte code/evaluator,
CRC, filenames/nhãn, logit từng view và các cách gộp, temperature, metrics và percentile
latency từ100 samples. Không có test predictions. Evidence ở `validation/stage5_kaggle/`.

Dẫn đầu: T05_smoothing +I04_res288, macro-F1 val0.970583, top1 0.978006,
ECE0.096209, p95 batch1 15.4685ms. T00+5crop_prob đạt macro-F1 0.967756 nhưng
p95 70.1866ms. AMP tăng throughput batch32 nhưng chậm hơn FP32 batch1 trong phép đo.

Temperature ở224 của T05: T0.609263, ECE từ0.094831 xuống0.006076, accuracy không đổi.
Đây là kết quả trên val dùng để fit, không phải calibration holdout. Chặng6 chốt train
smoothing +FP32 single288, fit T trên val288 riêng mỗi seed rồi áp dụng test; baseline
CE+FP32 single224 không temperature. Không áp T của224 lên288. Chưa có số liệu test.
