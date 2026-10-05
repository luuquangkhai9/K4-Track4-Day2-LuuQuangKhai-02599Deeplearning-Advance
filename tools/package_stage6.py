"""Self-contained frozen final notebook:6 training runs and single test passes."""
import json
from pathlib import Path
from notebook_bundle import Notebook
ROOT=Path(__file__).resolve().parents[1];SUB=ROOT/'submissions/02599_LuuQuangKhai'
nb=Notebook()
nb.add('markdown',"""
# Chặng6 — Final/baseline,3 seed, test sau khi chốt cấu hình
Chỉ gắn dataset ảnh hiện có. Bật GPU+Internet; **Save Version → Save & Run All**.
Notebook tự chứa code/test/CSV và bằng chứng chọn cấu hình; không cần output chặng4/5.

**F01:** ConvNeXt-Tiny fine-tune + label smoothing0.1, train224, inference FP32 288 +temperature
fit trên val288 riêng mỗi seed. **T00:** CE +1view FP32 224, T1.
Train lại cả hai với seed0,1,2,10 epoch/run, batch32. Không dùng test để chọn cấu hình.

Sáu run train và calibration/latency trên val hoàn tất trước vòng test.
Mỗi run test một lượt; raw logits dùng chung cho calibrated/uncalibrated và được lưu để recovery.
Sau khi có test, không đổi recipe hoặc xoá receipt để chạy lại.
""")
nb.add('code',"""
from pathlib import Path
import sys,subprocess,shutil,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
IMAGES_DIR=Path('/kaggle/input/datasets/luuquangkhai/data-labd2/images')
SESSION_BUDGET_HOURS=6.0
PREVIOUS_OUTPUT=None  # deepweeds_stage6/outputs từ phiên CHẶNG6 trước nếu resume
WORK=Path('/kaggle/working/deepweeds_stage6');OUT=WORK/'outputs'
OUT.mkdir(parents=True,exist_ok=True)
if PREVIOUS_OUTPUT:
    previous=Path(PREVIOUS_OUTPUT)
    assert (previous/'frozen_recipe.json').is_file() and (previous/'suite_config.json').is_file(), 'Chọn output chặng6'
    assert not (OUT/'runs').exists(),'Đặt PREVIOUS_OUTPUT=None nếu đã copy trong session này'
    shutil.copytree(previous,OUT,dirs_exist_ok=True)
assert (IMAGES_DIR/'20160928-140314-0.jpg').is_file(),'Kiểm tra IMAGES_DIR'
sys.path.insert(0,str(WORK/'code'))
print('Images:',IMAGES_DIR,'| Output:',OUT)
""")
files=[SUB/'STAGE6.md',SUB/'stage6_selection.json',*sorted((SUB/'code').glob('*.py')),
       SUB/'tests/test_pipeline.py',SUB/'tests/test_stage5.py',SUB/'tests/test_stage6.py',
       *sorted((SUB/'labels').glob('*.csv')),SUB/'eda/kaggle/stage1_summary.json',
       SUB/'validation/stage5_kaggle/verification.json',SUB/'validation/stage5_kaggle/inference_ranked.csv']
manifest=nb.embed(files,SUB,'stage6_manifest.json')
nb.add('code',"""
import importlib.metadata
try:version=importlib.metadata.version('timm')
except importlib.metadata.PackageNotFoundError:version=None
if version!='1.0.30':subprocess.check_call([sys.executable,'-m','pip','install','timm==1.0.30'])
import torch,torchvision,timm
assert torch.cuda.is_available(),'Bật GPU trong Settings'
print('torch',torch.__version__,'torchvision',torchvision.__version__,'timm',timm.__version__,'GPU',torch.cuda.get_device_name(0))
with (OUT/'pip-freeze.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
for test in ('test_pipeline.py','test_stage5.py','test_stage6.py'):
    result=subprocess.run([sys.executable,str(WORK/'tests'/test)],capture_output=True,text=True)
    log=result.stdout+'\\n'+result.stderr;(OUT/(test+'.log')).write_text(log,encoding='utf-8');print(log)
    assert result.returncode==0,f'Test failed: {test}; gửi log, chưa train'
from dataset import load_split,verified_stage1,check_split
LABELS_DIR=WORK/'labels';RECEIPT=WORK/'eda/kaggle/stage1_summary.json'
reuse=verified_stage1(RECEIPT,IMAGES_DIR,LABELS_DIR,0)
check_split(*load_split(LABELS_DIR),IMAGES_DIR,verify_files=not reuse)
print('Dùng lại biên nhận kiểm tra ảnh:',reuse)
shutil.copyfile(WORK/'validation/stage5_kaggle/verification.json',OUT/'stage5_verification.json')
shutil.copyfile(WORK/'validation/stage5_kaggle/inference_ranked.csv',OUT/'stage5_inference_ranked.csv')
""")
nb.add('code',"""
from stage6 import make_configs,validate_configs,run_suite,freeze
selection=json.loads((WORK/'stage6_selection.json').read_text())
configs=make_configs(IMAGES_DIR,LABELS_DIR,OUT,receipt=RECEIPT,device='cuda:0')
validate_configs(configs)
assert configs[0].backbone in timm.list_pretrained(configs[0].backbone)
print(json.dumps(selection,indent=2))
for cfg in configs:print(cfg.exp_id,'seed',cfg.seed,'loss',cfg.loss,'train224; epochs',cfg.epochs)
print('Frozen recipe SHA256:',freeze(configs,selection,OUT,WORK/'code'))
""")
nb.add('markdown',"""
## Chạy cả chặng6
Mặc định ngân sách mềm6h. Tốc độ chặng4 gợi ý khoảng75 phút train+val224, cộng setup,
lưu checkpoint và các bước eval cuối; đây không phải cam kết thời gian Kaggle.
ZIP nhỏ cập nhật sau từng run, best/last giữ trong output notebook.

Vòng test chỉ bắt đầu sau đủ6 train +6 val calibration/latency. test_started ghi trước khi đọc ảnh.
Ngắt giữa test mà chưa có logit cache đầy đủ sẽ dừng, gửi traceback/receipt; không xoá marker.
Nếu ngắt sau khi cache đã đủ, resume chỉ xuất kết quả từ cache, không chạy model test lại.
""")
nb.add('code',"""
report=run_suite(configs,selection,OUT,WORK/'code','/kaggle/working/stage6_results.zip',
                 budget_seconds=SESSION_BUDGET_HOURS*3600)
""")
nb.add('code',"""
import pandas as pd
from IPython.display import display
if (OUT/'final_mean_std.csv').exists():display(pd.read_csv(OUT/'final_mean_std.csv'))
print(json.dumps(report,indent=2))
print('Gửi lại /kaggle/working/stage6_results.zip; giữ output có best.pt/last.pt và test receipts.')
if report['status']!='complete':print('Chưa hoàn tất; resume đúng notebook từ output CHẶNG6, không đổi recipe.')
else:
    print('Đủ3 seed final và baseline, score/grade đã chạy. Tiếp theo tổng hợp bài nộp chặng7.')
    print((OUT/'official/grade.txt').read_text())
""")
nb.write(SUB/'code/kaggle_stage6.ipynb')
(SUB/'stage6_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Created',len(nb.cells),'cells;',len(manifest),'embedded files')
