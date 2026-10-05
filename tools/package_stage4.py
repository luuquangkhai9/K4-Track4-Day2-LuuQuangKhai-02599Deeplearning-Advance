"""One self-contained stage-4 Kaggle notebook; no local training prerequisite."""
import json
from pathlib import Path
from notebook_bundle import Notebook
ROOT=Path(__file__).resolve().parents[1];SUB=ROOT/'submissions/02599_LuuQuangKhai'
nb=Notebook()
nb.add('markdown',"""
# Chặng 4 — Ablation ConvNeXt-Tiny, chạy hoàn toàn trên Kaggle
Gắn dataset ảnh hiện có. Bật GPU + Internet, dùng **Save Version → Save & Run All**.
Notebook tự chứa code, test, CSV và biên nhận EDA; không cần upload code ZIP hoặc checkpoint chặng3.

Chặng3 đã xác nhận: ConvNeXt-Tiny macro-F1 val **0.964603**, top1 **0.972579**, dẫn đầu5 backbone.
Chặng4 chạy T00 +6 ablation ba trục (khởi tạo/augmentation/loss) +1 kết hợp chọn từ val.
Cùng fold0, seed0,10 epoch, ảnh224, batch32. Test vẫn khóa; nhiều seed ở chặng6.
Các bảng bằng chứng chặng3 là kết quả đã kiểm tra; không có metric chặng4 điền sẵn.

Mặc định ngân sách mềm6 giờ cho vòng train. Có thể resume bằng đúng notebook từ output chặng4.
""")
nb.add('code',"""
from pathlib import Path
import sys,subprocess,shutil,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
IMAGES_DIR=Path('/kaggle/input/datasets/luuquangkhai/data-labd2/images')
SESSION_BUDGET_HOURS=6.0
PREVIOUS_OUTPUT=None  # outputs CHẶNG4 đã lưu nếu cần resume, không dùng chặng3
WORK=Path('/kaggle/working/deepweeds_stage4');OUT=WORK/'outputs'
OUT.mkdir(parents=True,exist_ok=True)
if PREVIOUS_OUTPUT:
    previous=Path(PREVIOUS_OUTPUT)
    assert (previous/'suite_identity.json').is_file() and (previous/'single_axis_configs.json').is_file(), 'Chọn outputs chặng4'
    assert not (OUT/'runs').exists(),'Đã có runs; đặt PREVIOUS_OUTPUT=None'
    shutil.copytree(previous,OUT,dirs_exist_ok=True)
assert (IMAGES_DIR/'20160928-140314-0.jpg').is_file(),'Kiểm tra IMAGES_DIR'
sys.path.insert(0,str(WORK/'code'))
print('Images:',IMAGES_DIR,'| Output:',OUT)
""")
files=[SUB/'STAGE4.md',*sorted((SUB/'code').glob('*.py')),
       *sorted((SUB/'tests').glob('*.py')),*sorted((SUB/'labels').glob('*.csv')),
       SUB/'eda/kaggle/stage1_summary.json',
       SUB/'validation/stage3_kaggle/verification.json',
       SUB/'validation/stage3_kaggle/backbones_ranked.csv']
manifest=nb.embed(files,SUB,'stage4_manifest.json')
nb.add('code',"""
import importlib.metadata
try:version=importlib.metadata.version('timm')
except importlib.metadata.PackageNotFoundError:version=None
if version!='1.0.30':subprocess.check_call([sys.executable,'-m','pip','install','timm==1.0.30'])
import torch,torchvision,timm
assert torch.cuda.is_available(),'Bật GPU trong Settings'
print('torch',torch.__version__,'torchvision',torchvision.__version__,'timm',timm.__version__)
print('GPU:',torch.cuda.get_device_name(0))
with (OUT/'pip-freeze.txt').open('w') as f:
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
""")
nb.add('code',"""
# Chạy kiểm tra trong Kaggle, không yêu cầu máy local có GPU hoặc chạy train local.
for test in ('test_pipeline.py','test_stage3.py','test_stage4.py'):
    result=subprocess.run([sys.executable,str(WORK/'tests'/test)],capture_output=True,text=True)
    log=result.stdout+'\\n'+result.stderr
    (OUT/(test+'.log')).write_text(log,encoding='utf-8')
    print(log)
    assert result.returncode==0,f'Test failed: {test}; gửi log, chưa train'
from dataset import load_split,check_split,verified_stage1
LABELS_DIR=WORK/'labels';RECEIPT=WORK/'eda/kaggle/stage1_summary.json'
reuse=verified_stage1(RECEIPT,IMAGES_DIR,LABELS_DIR,0)
check_split(*load_split(LABELS_DIR),IMAGES_DIR,verify_files=not reuse)
print('Dùng lại biên nhận kiểm tra ảnh chặng1:',reuse)
# Bằng chứng chọn backbone đã kiểm tra tại local; ghi vào output chặng4.
shutil.copyfile(WORK/'validation/stage3_kaggle/verification.json',OUT/'stage3_verification.json')
shutil.copyfile(WORK/'validation/stage3_kaggle/backbones_ranked.csv',OUT/'stage3_backbones_ranked.csv')
""")
nb.add('markdown',"""
## Kế hoạch cố định
T00: pretrained fine-tune + crop/flip + CE. T01 scratch; T02 chỉ head.
T03 thêm RandAugment; T04 thêm CutMix alpha1. T05 label smoothing0.1; T06 focal gamma2.
Mỗi ablation thay một yếu tố; nền T00 giữ cố định. T07 chọn từng giá trị nếu macro-F1 val hơn T00;
hòa/kém giữ nền. Nếu tất cả giữ nền, T07 được ghi là lặp nền.

Mọi run10 epoch, batch32,224px, LR backbone1e-4/head1e-3, AdamW, warmup1+cosine, AMP.
Scratch giữ cùng ngân sách, không giả định đã hội tụ. Không tiếp tục model từ thí nghiệm khác.
""")
nb.add('code',"""
from stage4 import make_configs,validate_plan,run_suite
from dataclasses import asdict
configs=make_configs(IMAGES_DIR,LABELS_DIR,OUT,receipt=RECEIPT,device='cuda:0')
validate_plan(configs)
assert configs[0].backbone in timm.list_pretrained(configs[0].backbone)
for cfg in configs:
    print(cfg.exp_id,'init=',cfg.init,'aug=',cfg.aug,'mix=',cfg.mix,'loss=',cfg.loss)
""")
nb.add('markdown',"""
## Chạy hàng loạt và xuất kết quả
Cell sau chạy tối đa8 thí nghiệm. Lưu last.pt sau mỗi epoch, ZIP nhỏ sau mỗi run và khi lỗi.
Thời gian B02 chặng3 gợi ý khoảng1.6 giờ train+val cho8 run nếu tốc độ tương tự; đây chưa gồm setup
và không bảo đảm thời gian Kaggle. Ngân sách mềm kiểm tra ở ranh giới epoch.
Nếu chưa xong, giữ output checkpoint để resume bằng PREVIOUS_OUTPUT.
""")
nb.add('code',"""
report=run_suite(configs,OUT,'/kaggle/working/stage4_results.zip',WORK/'code',
                 budget_seconds=SESSION_BUDGET_HOURS*3600)
""")
nb.add('code',"""
import pandas as pd
from IPython.display import display,Image
columns=['exp_id','axis','status','val_macro_f1','val_top1','delta_macro_f1_vs_T00','best_epoch']
table=pd.read_csv(OUT/'training.csv')
display(table[[c for c in columns if c in table]])
if (OUT/'training_ablation.png').exists():display(Image(filename=str(OUT/'training_ablation.png')))
print(json.dumps(report,indent=2))
print('Gửi lại: /kaggle/working/stage4_results.zip')
print('Giữ output deepweeds_stage4/outputs/runs có best.pt và last.pt cho chặng5/resume.')
if report['status']!='complete':print('CHƯA ĐỦ8 RUN: lưu output rồi resume đúng notebook/config.')
else:print('Đủ8 RUN; chọn ứng viên chặng5 sau khi kiểm tra val. Test chưa được đánh giá.')
""")
nb.write(SUB/'code/kaggle_stage4.ipynb')
(SUB/'stage4_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Created stage4 notebook:',len(nb.cells),'cells;',len(manifest),'embedded files')
