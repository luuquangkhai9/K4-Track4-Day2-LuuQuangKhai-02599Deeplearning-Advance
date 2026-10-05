"""Generate a self-contained inference notebook; user's stage4 checkpoint output is input."""
import json
from pathlib import Path
from notebook_bundle import Notebook
ROOT=Path(__file__).resolve().parents[1];SUB=ROOT/'submissions/02599_LuuQuangKhai'
nb=Notebook()
nb.add('markdown',"""
# Chặng5 — Suy luận, TTA, resolution, temperature và AMP
Gắn **dataset ảnh + output notebook chặng4** làm input. Bật GPU và Internet.
Save Version → Save & Run All. Không cần ZIP code, không train lại, không đánh giá test.

Cần best.pt của T00 và T05 từ output chặng4. stage4_results.zip không chứa trọng số.
16 cấu hình:8 phương pháp ×2 checkpoint. Đo thực tế batch1/32,10 warmup +100 lượt.
Calibration fit trên fullval và báo rõ metrics cũng đo trên tập fit; không phải holdout độc lập.
""")
nb.add('code',"""
from pathlib import Path
import sys,subprocess,shutil,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
IMAGES_DIR=Path('/kaggle/input/datasets/luuquangkhai/data-labd2/images')
STAGE4_OUTPUT=None  # nếu cần: '/kaggle/input/.../deepweeds_stage4/outputs'
PREVIOUS_OUTPUT=None  # outputs CHẶNG5 đã lưu để resume
WORK=Path('/kaggle/working/deepweeds_stage5');OUT=WORK/'outputs'
OUT.mkdir(parents=True,exist_ok=True)
if STAGE4_OUTPUT is None:
    # Tìm trong output input, không quét ảnh dataset.
    candidates=[]
    for checkpoint in Path('/kaggle/input').glob('**/runs/T05_smoothing/seed0/best.pt'):
        source=checkpoint.parents[3]
        if (source/'runs/T00/seed0/best.pt').is_file() and (source/'single_axis_configs.json').is_file():
            candidates.append(source)
    assert len(candidates)==1,f'Cần add output notebook chặng4; nếu nhiều nguồn, điền STAGE4_OUTPUT. Found: {candidates}'
    STAGE4_OUTPUT=candidates[0]
STAGE4_OUTPUT=Path(STAGE4_OUTPUT)
assert (STAGE4_OUTPUT/'runs/T00/seed0/best.pt').is_file()
assert (STAGE4_OUTPUT/'runs/T05_smoothing/seed0/best.pt').is_file()
if PREVIOUS_OUTPUT:
    previous=Path(PREVIOUS_OUTPUT)
    assert (previous/'suite_identity.json').is_file() and (previous/'stage5_summary.json').is_file()
    assert not (OUT/'methods').exists(),'Đặt PREVIOUS_OUTPUT=None khi đã copy trong session này'
    shutil.copytree(previous,OUT,dirs_exist_ok=True)
assert (IMAGES_DIR/'20160928-140314-0.jpg').is_file(),'Kiểm tra IMAGES_DIR'
sys.path.insert(0,str(WORK/'code'))
print('Checkpoint source:',STAGE4_OUTPUT,'| Output:',OUT)
""")
files=[SUB/'STAGE5.md',*sorted((SUB/'code').glob('*.py')),
       SUB/'tests/test_pipeline.py',SUB/'tests/test_stage5.py',*sorted((SUB/'labels').glob('*.csv')),
       SUB/'eda/kaggle/stage1_summary.json',SUB/'validation/stage4_kaggle/verification.json']
# Embed genuine stage4 reference logits for I00 checkpoint reproduction.
for exp in ('T00','T05_smoothing'):
    files.append(SUB/f'validation/stage4_kaggle/runs/{exp}/seed0/val_logits.npz')
manifest=nb.embed(files,SUB,'stage5_manifest.json')
nb.add('code',"""
import importlib.metadata
try:version=importlib.metadata.version('timm')
except importlib.metadata.PackageNotFoundError:version=None
if version!='1.0.30':subprocess.check_call([sys.executable,'-m','pip','install','timm==1.0.30'])
import torch,torchvision,timm
assert torch.cuda.is_available(),'Bật GPU'
print('torch',torch.__version__,'torchvision',torchvision.__version__,'timm',timm.__version__,'GPU',torch.cuda.get_device_name(0))
with (OUT/'pip-freeze.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
for test in ('test_pipeline.py','test_stage5.py'):
    result=subprocess.run([sys.executable,str(WORK/'tests'/test)],capture_output=True,text=True)
    log=result.stdout+'\\n'+result.stderr;(OUT/(test+'.log')).write_text(log,encoding='utf-8');print(log)
    assert result.returncode==0,f'Test failed: {test}; gửi log trước khi suy luận'
from dataset import verified_stage1,check_split,load_split
LABELS_DIR=WORK/'labels';RECEIPT=WORK/'eda/kaggle/stage1_summary.json'
reuse=verified_stage1(RECEIPT,IMAGES_DIR,LABELS_DIR,0)
check_split(*load_split(LABELS_DIR),IMAGES_DIR,verify_files=not reuse)
print('Dùng lại biên nhận ảnh chặng1:',reuse)
REFERENCE=WORK/'reference';REFERENCE.mkdir(exist_ok=True)
for exp in ('T00','T05_smoothing'):
    shutil.copyfile(WORK/f'validation/stage4_kaggle/runs/{exp}/seed0/val_logits.npz',REFERENCE/f'{exp}_val_logits.npz')
shutil.copyfile(WORK/'validation/stage4_kaggle/verification.json',OUT/'stage4_verification.json')
""")
nb.add('markdown',"""
## Chạy16 phép so sánh
I00 single224; I01 hflip K2; I02/I03 5crop xác suất/logit; I04 resolution256/288;
I07 temperature từ val; I08 AMP. Mỗi run dùng toàn bộ3501 ảnh val.
Kết quả từng phương pháp lưu vào ZIP để resume. Độ trễ bao gồm view/forward/aggregation,
loại trừ đọc ảnh/resize/normalize/copy CPU→GPU. Không so tuyệt đối với forward-only chặng3.
""")
nb.add('code',"""
from stage5 import run_suite
report=run_suite(STAGE4_OUTPUT,IMAGES_DIR,LABELS_DIR,OUT,WORK/'code',
                 '/kaggle/working/stage5_results.zip',REFERENCE)
""")
nb.add('code',"""
import pandas as pd
from IPython.display import display,Image
display(pd.read_csv(OUT/'inference.csv'))
if (OUT/'inference_tradeoff.png').exists():display(Image(filename=str(OUT/'inference_tradeoff.png')))
print(json.dumps(report,indent=2))
print('Gửi lại /kaggle/working/stage5_results.zip; giữ output chặng4 có checkpoint.')
print('Chưa chốt final; sẽ đọc kết quả val, calibration và p95 trước chặng6 nhiều seed.')
""")
nb.write(SUB/'code/kaggle_stage5.ipynb')
(SUB/'stage5_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Created',len(nb.cells),'cells;',len(manifest),'embedded files')
