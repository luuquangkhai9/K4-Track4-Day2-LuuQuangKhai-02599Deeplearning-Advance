"""Generate and package the stage-2 Kaggle notebook, code, tests and labels."""
import base64
import hashlib
import zlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions/02599_LuuQuangKhai"
cells = []


def cell(kind, source):
    value = {"cell_type": kind, "metadata": {}, "source": source.strip() + "\n"}
    if kind == "code":
        compile(value["source"], "stage2 notebook cell", "exec")
        value.update(execution_count=None, outputs=[])
    cells.append(value)


cell("markdown", """
# DeepWeeds — Chặng 2: kiểm tra pipeline trước khi train hàng loạt
Notebook tự chứa code, test, CSV nhãn và biên nhận chặng 1.
Chỉ gắn dataset ảnh hiện tại; không cần upload ZIP hoặc gắn dataset code riêng.
Các cell đầu tạo file trong working; không cần chạy EDA lại.
Bật GPU + Internet. Notebook dùng **một T4**; không tự cộng VRAM của hai GPU.

Thứ tự: xác nhận môi trường → kiểm tra CPU → overfit 9 ảnh train → thử resume trên tập nhỏ
→ train một epoch toàn bộ train/val → đóng gói output. Không đọc ảnh test, không đánh giá test.
Các lượt S01/S02 là kiểm tra kỹ thuật, không đưa vào bảng so sánh backbone.
Sau khi sửa đường dẫn nếu cần, dùng **Save Version → Save & Run All**.
""")
cell("code", """
from pathlib import Path
import sys, subprocess, shutil, zipfile, json, os

IMAGES_DIR = Path('/kaggle/input/datasets/luuquangkhai/data-labd2/images')
# Chỉ điền khi tiếp tục một phiên cũ: đường dẫn tới thư mục outputs của phiên đã lưu.
PREVIOUS_OUTPUT = None
WORK = Path('/kaggle/working/deepweeds_stage2')
OUT = WORK / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

if PREVIOUS_OUTPUT:
    previous = Path(PREVIOUS_OUTPUT)
    assert previous.is_dir() and (previous / 'runs').is_dir(), 'Chọn đúng thư mục outputs cũ'
    # Chỉ khôi phục trước lần chạy đầu, tránh ghi đè checkpoint mới trong session.
    assert not (OUT / 'runs').exists(), 'outputs/runs đã có; đặt PREVIOUS_OUTPUT=None và chạy tiếp'
    shutil.copytree(previous, OUT, dirs_exist_ok=True)
assert (IMAGES_DIR / '20160928-140314-0.jpg').is_file(), 'Kiểm tra đường dẫn ảnh'
sys.path.insert(0, str(WORK / 'code'))
print('Code:', WORK, '\\nImages:', IMAGES_DIR)
""")
# Embed readable Python modules; preserve original bytes so existing checkpoints remain compatible.
payload_files = [SUB / "README.md", SUB / "STAGE2.md",
                 *sorted((SUB / "code").glob("*.py")), *sorted((SUB / "tests").glob("*.py")),
                 *sorted((SUB / "labels").glob("*.csv")), SUB / "eda/kaggle/stage1_summary.json"]
manifest = {p.relative_to(SUB).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in payload_files}
cell("markdown", """
## Code đi kèm notebook
Chạy các cell này một lần để tạo các module trong working. Code được hiển thị trực tiếp
để đọc/đối chiếu. CSV nhãn được nén trong notebook và khôi phục nguyên byte.
SHA256 kiểm tra nội dung nhúng; nếu muốn sửa module, sửa bản local và tạo lại notebook.
""")
cell("code", """
import hashlib, base64, zlib

def write_embedded(relative, payload, expected):
    assert hashlib.sha256(payload).hexdigest() == expected, f'Nội dung nhúng bị thay đổi: {relative}'
    target = WORK / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
""")
for p in payload_files:
    if p.suffix != ".py":
        continue
    raw = p.read_bytes()
    text = raw.decode("utf-8").replace("\r\n", "\n")
    delimiter = "'" * 3
    assert delimiter not in text, p
    crlf_lines = [i for i, line in enumerate(raw.split(b"\n")) if line.endswith(b"\r")]
    lines = text.split("\n")
    for i in crlf_lines:
        lines[i] += "\r"
    assert "\n".join(lines).encode("utf-8") == raw, p
    rel = p.relative_to(SUB).as_posix()
    source = f"# {rel}\n_embedded_source = r{delimiter}{text}{delimiter}\n"
    source += "_lines = _embedded_source.split('\\n')\n"
    source += f"for _line_index in {crlf_lines!r}:\n    _lines[_line_index] += '\\r'\n"
    source += f"write_embedded({rel!r}, '\\n'.join(_lines).encode('utf-8'), {manifest[rel]!r})\n"
    cell("code", source)
compressed = {p.relative_to(SUB).as_posix(): base64.b64encode(zlib.compress(p.read_bytes(), 9)).decode("ascii")
              for p in payload_files if p.suffix != ".py"}
cell("code", "# CSV gốc và tài liệu nhúng; không tải nhãn lại từ Internet.\n"
     + "_embedded_data = " + repr(compressed) + "\n"
     + "_manifest = " + repr(manifest) + "\n"
     + "for relative, encoded in _embedded_data.items():\n"
     + "    write_embedded(relative, zlib.decompress(base64.b64decode(encoded)), _manifest[relative])\n"
     + "(WORK / 'stage2_manifest.json').write_text(json.dumps(_manifest, indent=2), encoding='utf-8')\n"
     + "print('Đã khôi phục và kiểm tra SHA256:', len(_manifest), 'file. Không cần dataset code.')\n")

cell("code", """
import importlib.metadata
try:
    installed_timm = importlib.metadata.version('timm')
except importlib.metadata.PackageNotFoundError:
    installed_timm = None
if installed_timm != '1.0.30':
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'timm==1.0.30'])
import torch, torchvision, timm
assert torch.cuda.is_available(), 'Chặng 2 cần bật GPU trong Settings trước khi chạy'
print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| timm', timm.__version__)
print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print('Dùng:', torch.cuda.get_device_name(0), '| VRAM GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
with (OUT / 'pip-freeze.txt').open('w') as f:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
""")
cell("code", """
# Test code trong subprocess riêng; lỗi sẽ dừng trước khi train.
test_result = subprocess.run([sys.executable, str(WORK / 'tests/test_pipeline.py')],
                             capture_output=True, text=True)
test_log = test_result.stdout + '\\n' + test_result.stderr
(OUT / 'pipeline_tests.txt').write_text(test_log)
print(test_log)
assert test_result.returncode == 0, 'Pipeline test thất bại; gửi pipeline_tests.txt'

from dataset import load_split, check_split, verified_stage1
LABELS_DIR = WORK / 'labels'
RECEIPT = WORK / 'eda/kaggle/stage1_summary.json'
reuse = verified_stage1(RECEIPT, IMAGES_DIR, LABELS_DIR, 0)
split = check_split(*load_split(LABELS_DIR), IMAGES_DIR, verify_files=not reuse)
print('Dùng lại kiểm tra ảnh chặng 1:', reuse)
""")
cell("markdown", """
## Overfit một batch
Chín ảnh train cố định, mỗi lớp một ảnh, không augmentation khi tối ưu.
LR chẩn đoán 1e-3, không weight decay. Dừng khi CE < 0.1 và accuracy = 1 trên batch đó.
Nếu chưa đạt sau 200 bước, notebook dừng để kiểm tra; không tự coi là thành công.
Trọng số pretrained sẽ tải lần đầu, cần Internet.
""")
cell("code", """
from smoke_checks import run_checks
from IPython.display import display, Image
smoke = run_checks(IMAGES_DIR, LABELS_DIR, OUT / 'smoke', device='cuda:0')
print(smoke)
display(Image(filename=str(OUT / 'smoke/augmentation_check.png')))
display(Image(filename=str(OUT / 'smoke/overfit_curve.png')))
torch.cuda.empty_cache()
""")
cell("markdown", """
## Kiểm tra resume trên tập nhỏ
S02 dùng 18 ảnh train, 9 ảnh val, ảnh 64×64, scratch; chỉ là chẩn đoán.
Tổng kế hoạch giữ nguyên 2 epoch: lượt đầu dừng sau 1 epoch, lượt sau resume đến epoch 2.
Không tăng `epochs` của checkpoint đang chạy vì sẽ đổi lịch LR.
""")
cell("code", """
from dataclasses import replace
from train import Config, run, run_dir
common = dict(images_dir=str(IMAGES_DIR), labels_dir=str(LABELS_DIR),
              verified_summary=str(RECEIPT), out_dir=str(OUT / 'runs'),
              pred_dir=str(OUT / 'predictions'), device='cuda:0', num_workers=2)
resume_cfg = Config(exp_id='S02_resume', epochs=2, batch_size=9, img_size=64,
                    init='scratch', debug_train_per_class=2, debug_val_per_class=1,
                    stop_after_epochs=1, **common)
first = run(resume_cfg)
second = run(replace(resume_cfg, stop_after_epochs=None))
assert second['status'] == 'completed' and second['epochs_completed'] == 2
print('Resume check:', second)
torch.cuda.empty_cache()
""")
cell("markdown", """
## Một epoch toàn bộ train/val
S01: MobileNetV3 pretrained, 224×224, batch 32, AMP; warmup 0.1 epoch vì đây chỉ là smoke run 1 epoch.
Sau mỗi 50 batch có log tiến độ. Không thay số epoch này để biến thành thí nghiệm chính thức.
Chặng 3 sẽ có exp_id mới, cùng 10 epoch và warmup 1 epoch cho mọi backbone.
""")
cell("code", """
cfg = Config(exp_id='S01_full_epoch', backbone='mobilenetv3_large_100.ra_in1k',
             epochs=1, warmup_epochs=0.1, batch_size=32, img_size=224, **common)
result = run(cfg)
assert result['status'] == 'completed'
assert result['train_images'] == 10501 and result['val_images'] == 3501
print(json.dumps(result, indent=2))
display(Image(filename=str(run_dir(cfg) / 'curves.png')))
# Gọi lại phải bỏ qua run đã hoàn tất, không train thêm.
assert run(cfg)['status'] == 'completed'
""")
cell("code", """
results_zip = Path('/kaggle/working/stage2_results.zip')
checkpoints_zip = Path('/kaggle/working/stage2_checkpoints.zip')
with zipfile.ZipFile(results_zip, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.suffix not in ('.pt', '.tmp'):
            z.write(p, p.relative_to(OUT).as_posix())
    for p in sorted((WORK / 'code').glob('*.py')):
        z.write(p, 'code/' + p.name)
    z.write(WORK / 'stage2_manifest.json', 'stage2_manifest.json')
with zipfile.ZipFile(checkpoints_zip, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted((OUT / 'runs').rglob('*.pt')):
        z.write(p, p.relative_to(OUT).as_posix())
print('Gửi lại file kết quả nhỏ:', results_zip)
print('Giữ bản checkpoint để khôi phục:', checkpoints_zip)
print('PASS — chặng 2 trên Kaggle hoàn tất; chưa train thí nghiệm chính thức, chưa chạy test.')
""")

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
path = SUB / "code/kaggle_stage2.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
# Keep the ZIP as an optional legacy artifact; the notebook does not depend on it.
(SUB / "stage2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
with zipfile.ZipFile(ROOT / "kaggle_stage2_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in payload_files + [path, SUB / "stage2_manifest.json"]:
        z.write(p, p.relative_to(SUB).as_posix())
print(f"Created self-contained {path} ({len(cells)} cells); optional legacy ZIP refreshed")
