# DeepWeeds — 02599 Lưu Quang Khải

## Kết quả đã kiểm tra

Final **F01: ConvNeXt-Tiny + label smoothing0.1, một view FP32 288px +temperature fit trên val từng seed**. Mốc **T00: cùng backbone, CE, một view FP32 224px, không calibration**. Đã hoàn tất chặng1–7; số liệu từ Kaggle được đối chiếu bằng evaluator gốc.

| Test, ba seed0/1/2 | F01 mean ± std | T00 mean ± std |
|---|---|---|
| Macro-F1 | **0.96947 ± 0.00196** | 0.95860 ± 0.00148 |
| Top-1 | **97.6428% ± 0.0594 pp** | 96.7684% ± 0.0436 pp |
| ECE | **0.00663 ± 0.00040** | 0.00946 ± 0.00111 |

Std mẫu ddof1; pp là điểm phần trăm. Final p95 batch1 lớn nhất qua ba seed **17.92ms trên Tesla T4**, loại decode/resize/normalize/CPU→GPU. `official/grade_I.json` đề xuất **20/20 phần I**, không phải điểm toàn bài.

**Giới hạn:** T07 cùng recipe/seed T05, chưa thử tương tác nhiều yếu tố thật. Báo cáo ghi rõ, không nhận điểm cho phần chưa thử. Không tìm cấu hình mới sau test. ZIP kết quả nhỏ không chứa trọng số; audit xác nhận hashes/receipts/logits/predictions, chưa trực tiếp nạp lại checkpoint. Giữ trọng số trong Kaggle saved output.

## Sản phẩm

- [results.xlsx](results.xlsx): đúng7 sheet Summary, Final, Backbones, Training, Inference, PerClass, Latency; nguồn từng dòng và công thức mean/std/delta.
- [report.md](report.md): thiết lập, kết quả, ma trận nhầm lẫn, ảnh lỗi và hạn chế.
- `curves/`:19 curve B/T/F +2 curve pipeline +1 overfit. Manifest ánh xạ21 run theo epoch tới history/config; overfit từ stage2/smoke.
- `predictions/`: test/val mọi seed final/baseline và F01_uncal, kèm screening. T00 seed0 ở đây là chặng6; snapshot chặng4 giữ riêng trong validation.
- `code/`, `tests/`, `labels/`: implementation, notebook, evaluator nguyên byte, kiểm tra các phần dễ sai, CSV gốc.
- `official/`, `validation/`, `figures/`, `confusions/`: score/grade, log/config/environment/logits/receipt, audit và ảnh phân tích.

Không đóng gói dataset ảnh, checkpoint lớn, cache thư viện hoặc .git. Không commit dataset/checkpoint vào git.

## Notebook và chạy lại

Notebook đã chạy: **[day2-lab-phase2 trên Kaggle](https://www.kaggle.com/code/luuquangkhai/day2-lab-phase2)**. Nếu private, cần quyền xem. Có thể import các notebook lưu trong code/ vào tài khoản khác. Chặng2–6 giữ output đã trả trong source.

| Chặng | Notebook | Input/mục đích |
|---|---|---|
| 1 | code/kaggle_stage1.ipynb | Ảnh +bundle code chặng1; EDA; stage1.py có CLI |
| 2 | code/kaggle_stage2.ipynb | Chỉ ảnh, code tự chứa; overfit,1epoch,resume |
| 3 | code/kaggle_stage3.ipynb | Chỉ ảnh;5 backbone seed0 |
| 4 | code/kaggle_stage4.ipynb | Chỉ ảnh;3 trục ablation |
| 5 | code/kaggle_stage5.ipynb | Ảnh +output chặng4 có best.pt T00/T05;16 cấu hình suy luận |
| 6 | code/kaggle_stage6.ipynb | Chỉ ảnh;train mới final/baseline3 seed, freeze rồi test |
| 7 | Audit/tổng hợp | Chỉ đọc kết quả đã lưu, không chạy lại model trên test |

Dataset ảnh của phiên đã chạy: `/kaggle/input/datasets/luuquangkhai/data-labd2/images`. Bật **GPU+Internet**, giữ recipe, **Save Version → Save & Run All**. Chặng6 là đường tái lập final ngắn nhất, không cần checkpoint cũ. Code dùng một T4/cuda:0;10epoch/run,batch32. Ngân sách mềm6h không phải cam kết hạn mức Kaggle.

Resume lần chạy dở: add saved output đúng chặng, đặt PREVIOUS_OUTPUT tới deepweeds_stageX/outputs. Giữ nguyên code/config và frozen_recipe/test receipts. Train resume ở ranh giới epoch. Test đã có cache chỉ xuất lại CSV/metrics; ngắt giữa test trước cache đầy đủ sẽ dừng, không tự rerun. Không xoá marker hoặc đổi recipe sau khi xem test. Một lần tái lập độc lập ở output mới phải ghi là lần chạy mới, không chọn bộ test tốt hơn để thay bài đã chốt.

## Môi trường và recipe

Python3.13.15, torch2.11.0+cu128, torchvision0.26.0+cu128, CUDA12.8, timm1.0.30, numpy2.1.3, pandas2.3.3, matplotlib3.10.0, Pillow12.3.0. Snapshot đầy đủ `validation/stage6_kaggle/pip-freeze.txt`. Kaggle notebook giữ torch/torchvision có sẵn và cài timm nếu thiếu. Local kiểm tra evaluator chỉ cần numpy/pandas, không cần GPU.

Original fold0:train10501/val3501/test3507. Fine-tune pretrained; head Normal(0,0.01)/bias0. RandomResizedCrop bicubic224+horizontal flip; AdamW LR backbone1e-4/head1e-3, wd0.05 trừ norm/bias; warmup1+cosine, AMP train, không EMA/sampler/mix final. Chọn epoch bằng macro-F1 val224, hòa epoch sớm hơn. Seed0/1/2 điều khiển shuffle/augmentation/head. F01 val/test288 resize329 center crop, T fit trên val288 mỗi seed; T00 resize256 center crop224,T1. Hashes/recipe nằm trong stage6_selection.json và validation/stage6_kaggle/frozen_recipe.json.

Giữ bất nhất nguồn `20170714-110407-3.jpg`: train Label0, master Label1; không sửa CSV. Chia ngẫu nhiên không đảm bảo độc lập địa điểm/mùa; test fold0 chưa chứng minh ngoài miền.

## Kiểm tra lại từ predictions, không chạy model

Từ thư mục bài nộp, môi trường có numpy/pandas:

```powershell
python code/eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv labels/test_subset0.csv --labels labels/labels.csv --tag F01
python code/eval.py score --pred "predictions/T00_seed*_test.csv" --test-csv labels/test_subset0.csv --labels labels/labels.csv --tag T00
python code/eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" --uncal "predictions/F01_uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" --test-csv labels/test_subset0.csv --val-csv labels/val_subset0.csv --labels labels/labels.csv --latency-p95-ms 17.92099714948563 --latency-method proper
python code/audit_submission.py
```

Windows console cũ: đặt `$env:PYTHONUTF8='1'`. Score/grade chỉ đọc CSV, không train, không fit trên test. Local audit chặng7 kiểm tra predictions/NPZ, mean/std ddof1, calibrated/uncal dùng chung logits, freeze/test receipts và latency samples. Workbook đã recalculation, thử thay đổi một seed rồi khôi phục, scan không có lỗi công thức và render từng sheet. Chưa kiểm tra trong Excel native; số xlsx khớp evaluator trong tolerance1e-7, hiển thị4 chữ số nhưng giữ giá trị đầy đủ.

## Nguồn và kiểm tra gói nộp

validation/stage2_kaggle:pipeline/smoke; stage3_kaggle:B01…B05; stage4_kaggle:T00…T07; stage5_kaggle:inference/latency; stage6_kaggle:sáu run final/baseline và raw test cache; stage7_audit:score/grade tính lại, nguồn xây workbook và audit. Fixtures/local smoke không được coi là thí nghiệm DeepWeeds GPU.

submission_manifest.json ghi SHA256 mọi file trong gói. audit_submission.py kiểm tra manifest nếu có. Gói `02599_LuuQuangKhai_submission.zip` nằm cạnh thư mục bài làm. Giữ output Kaggle có checkpoint để tiếp tục nghiên cứu; không thay đổi final hoặc đánh giá thêm cấu hình trên test bài nộp này.
