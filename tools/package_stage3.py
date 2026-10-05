"""Generate stage 3 as ONE notebook; images remain in the existing Kaggle dataset."""
import json
from pathlib import Path
from notebook_bundle import Notebook

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions/02599_LuuQuangKhai"
nb = Notebook()
nb.add("markdown", """
# Chặng 3 — So sánh 5 backbone trên DeepWeeds
Chỉ gắn dataset ảnh đang dùng. Notebook tự chứa code, test, nhãn và biên nhận EDA.
GPU + Internet phải bật. Chạy **Save Version → Save & Run All**.

5 model chạy tuần tự trên cuda:0: ResNet50, ConvNeXt-Tiny, DeiT-Small, EfficientNet-B0, MobileNetV3.
Cùng seed 0, fold 0, 224×224, batch 32, 10 epoch, AdamW, CE, AMP và warmup 1 epoch.
Đây là sàng backbone trên VAL; TEST vẫn khóa. Chưa có kết quả nào được điền sẵn.

Mặc định ngân sách mềm 6 giờ cho vòng thí nghiệm; đây không phải cam kết về hạn mức Kaggle.
Nếu chưa xong, lưu output rồi resume bằng đúng notebook và cấu hình. Không giảm riêng
batch/epoch/LR cho một model. Lỗi sẽ xuất kết quả đã có và dừng để kiểm tra.
""")
nb.add("code", """
from pathlib import Path
import sys, subprocess, shutil, json, os

IMAGES_DIR = Path('/kaggle/input/datasets/luuquangkhai/data-labd2/images')
SESSION_BUDGET_HOURS = 6.0  # chỉnh theo hạn mức thực tế; không phải thời gian dự báo
PREVIOUS_OUTPUT = None     # thư mục outputs từ lần chạy CHẶNG 3 đã lưu, nếu cần resume
WORK = Path('/kaggle/working/deepweeds_stage3')
OUT = WORK / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
if PREVIOUS_OUTPUT:
    previous = Path(PREVIOUS_OUTPUT)
    assert (previous / 'suite_identity.json').is_file(), 'Chọn outputs chặng 3, không dùng chặng 2'
    assert not (OUT / 'runs').exists(), 'Đã có runs; đặt PREVIOUS_OUTPUT=None để tránh ghi đè'
    shutil.copytree(previous, OUT, dirs_exist_ok=True)
assert (IMAGES_DIR / '20160928-140314-0.jpg').is_file(), 'Kiểm tra IMAGES_DIR'
sys.path.insert(0, str(WORK / 'code'))
print('Images:', IMAGES_DIR, '| Output:', OUT)
""")
files = [SUB / "STAGE3.md", *sorted((SUB / "code").glob("*.py")),
         *sorted((SUB / "tests").glob("*.py")), *sorted((SUB / "labels").glob("*.csv")),
         SUB / "eda/kaggle/stage1_summary.json"]
manifest = nb.embed(files, SUB, "stage3_manifest.json")
nb.add("code", """
import importlib.metadata
try:
    version = importlib.metadata.version('timm')
except importlib.metadata.PackageNotFoundError:
    version = None
if version != '1.0.30':
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'timm==1.0.30'])
import torch, torchvision, timm
assert torch.cuda.is_available(), 'Bật GPU trong Settings trước khi chạy'
print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| timm', timm.__version__)
print('GPU dùng:', torch.cuda.get_device_name(0))
with (OUT / 'pip-freeze.txt').open('w') as f:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
""")
nb.add("code", """
# Kiểm tra source và kiến trúc trên CPU trước khi dùng GPU lâu.
for test in ('test_pipeline.py', 'test_stage3.py'):
    result = subprocess.run([sys.executable, str(WORK / 'tests' / test)], capture_output=True, text=True)
    log = result.stdout + '\\n' + result.stderr
    (OUT / (test + '.log')).write_text(log)
    print(log)
    assert result.returncode == 0, f'Test failed: {test}; gửi log, chưa train'

from dataset import load_split, check_split, verified_stage1
LABELS_DIR = WORK / 'labels'
RECEIPT = WORK / 'eda/kaggle/stage1_summary.json'
reuse = verified_stage1(RECEIPT, IMAGES_DIR, LABELS_DIR, 0)
check_split(*load_split(LABELS_DIR), IMAGES_DIR, verify_files=not reuse)
print('Dùng lại xác nhận ảnh chặng 1:', reuse)
""")
nb.add("code", """
from stage3 import make_configs, run_suite
configs = make_configs(IMAGES_DIR, LABELS_DIR, OUT, receipt=RECEIPT, device='cuda:0')
for cfg in configs:
    assert cfg.backbone in timm.list_pretrained(cfg.backbone), cfg.backbone
    print(cfg.exp_id, cfg.backbone, 'epochs=', cfg.epochs, 'batch=', cfg.batch_size)
""")
nb.add("markdown", """
## Chạy hàng loạt
Cell này có thể chạy nhiều giờ. Sau mỗi 50 batch có log; sau mỗi epoch lưu `last.pt`.
Mỗi model hoàn tất được đo GMAC và độ trễ FP32 batch 1 (10 warmup, 100 lượt đo).
Kết quả nhỏ được đóng gói sau mỗi model, kể cả khi model sau lỗi.
Nếu thời gian còn ít, vòng chạy tạm dừng; trạng thái `incomplete` nghĩa là cần resume.
""")
nb.add("code", """
report = run_suite(configs, OUT, '/kaggle/working/stage3_results.zip', WORK / 'code',
                   budget_seconds=SESSION_BUDGET_HOURS * 3600)
""")
nb.add("code", """
import pandas as pd
from IPython.display import display, Image
display(pd.read_csv(OUT / 'backbones.csv'))
if (OUT / 'backbone_tradeoff.png').exists():
    display(Image(filename=str(OUT / 'backbone_tradeoff.png')))
print(json.dumps(report, indent=2))
print('Gửi lại: /kaggle/working/stage3_results.zip')
print('Giữ output phiên có deepweeds_stage3/outputs/runs để bảo toàn best.pt và last.pt.')
if report['status'] != 'complete':
    print('CHƯA ĐỦ 5 BACKBONE: lưu output và resume cùng notebook, không đổi recipe.')
else:
    print('Đã có đủ 5 backbone. Chọn backbone bước tiếp theo sau khi đọc bảng VAL, không mở TEST.')
""")
path = SUB / "code/kaggle_stage3.ipynb"
nb.write(path)
(SUB / "stage3_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Created {path}: {len(nb.cells)} cells, {path.stat().st_size} bytes")
