"""Build the stage-1 Kaggle notebook and upload archive, without image data."""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions/02599_LuuQuangKhai"
cells = []


def cell(kind, text):
    item = {"cell_type": kind, "metadata": {}, "source": text.strip() + "\n"}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell("markdown", """
# DeepWeeds — Chặng 1: chuẩn bị và EDA
Gắn hai dataset: ảnh DeepWeeds và kaggle_stage1_code.zip. Chạy lần lượt từ trên xuống.
Chặng này không train và không tính metric test. CPU đủ cho EDA; GPU chỉ cần kiểm tra sẵn.
Kết quả tải về: `/kaggle/working/stage1_results.zip`.
""")
cell("code", """
import importlib.util
import subprocess
import sys
packages = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib", "PIL": "pillow"}
missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
if not torch.cuda.is_available():
    print("CPU vẫn chạy được chặng 1. Cần bật GPU trước chặng 2.")
""")
cell("code", """
from pathlib import Path
import shutil
import zipfile
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/deepweeds_stage1")
WORK.mkdir(parents=True, exist_ok=True)
sources = list(INPUT.rglob("stage1.py"))
archives = list(INPUT.rglob("kaggle_stage1_code.zip"))
if len(sources) == 1:
    source = sources[0].parent.parent
    shutil.copytree(source, WORK, dirs_exist_ok=True)
elif not sources and len(archives) == 1:
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(WORK)
else:
    raise RuntimeError(f"Cần đúng một nguồn code. Tìm thấy: {sources}, {archives}")
assert (WORK / "code/stage1.py").is_file()
sys.path.insert(0, str(WORK / "code"))
from stage1 import prepare_labels, run
print("Code:", WORK)
""")
cell("code", """
image_zips = list(INPUT.rglob("images.zip"))
image_roots = sorted({p.parent for p in INPUT.rglob("20160928-140314-0.jpg")})
if len(image_zips) == 1:
    IMAGES = image_zips[0]
elif not image_zips and len(image_roots) == 1:
    IMAGES = image_roots[0]
else:
    raise RuntimeError(f"Cần đúng một nguồn ảnh. ZIP: {image_zips}; thư mục: {image_roots}")
LABELS = prepare_labels(WORK / "labels")
OUT = WORK / "eda/kaggle"
print("Images:", IMAGES)
print("Labels:", LABELS)
""")
cell("markdown", """
## Kiểm tra và EDA
Cell tiếp theo đọc toàn bộ ảnh để kiểm tra hỏng file, đối chiếu nhãn, kiểm tra split.
Thống kê test chỉ dùng kiểm tra tính toàn vẹn dữ liệu. 27 ảnh minh họa lấy từ train.
Nếu kiểm tra báo lỗi, dừng tại đây và gửi traceback.
""")
cell("code", """
summary = run(IMAGES, LABELS, OUT)
assert summary["status"] == "PASS"
with (OUT / "pip-freeze.txt").open("w") as f:
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f, check=True)
""")
cell("code", """
from IPython.display import display, Image
display(Image(filename=str(OUT / "class_distribution.png")))
display(Image(filename=str(OUT / "train_samples.png")))
""")
cell("code", """
archive_path = Path("/kaggle/working/stage1_results.zip")
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            archive.write(p, "eda/" + p.relative_to(OUT).as_posix())
    for p in sorted(LABELS.glob("*.csv")):
        archive.write(p, "labels/" + p.name)
print("PASS — tải file này về và gửi lại:", archive_path)
""")
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
path = SUB / "code/kaggle_stage1.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
for c in cells:
    if c["cell_type"] == "code":
        compile(c["source"], "notebook cell", "exec")
with zipfile.ZipFile(ROOT / "kaggle_stage1_code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    files = [SUB / "README.md", *sorted((SUB / "code").glob("*.py")), path,
             *sorted((SUB / "labels").glob("*.csv"))]
    for file in files:
        archive.write(file, file.relative_to(SUB).as_posix())
print("Created notebook and kaggle_stage1_code.zip")
