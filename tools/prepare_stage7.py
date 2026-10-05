"""Collect verified experimental tables and inspect mistakes from saved predictions only."""
import json,sys,shutil,zipfile,io
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];SUB=ROOT/'submissions/02599_LuuQuangKhai'
sys.path.insert(0,str(SUB/'code'))
from eval import CLASS_NAMES,read_pred,compute_metrics
V=SUB/'validation';figs=SUB/'figures';figs.mkdir(exist_ok=True)

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def section(headers,rows,note,widths):return {'headers':headers,'rows':rows,'note':note,'widths':widths}
back=pd.read_csv(V/'stage3_kaggle/backbones_ranked.csv')
training=pd.read_csv(V/'stage4_kaggle/training.csv').fillna('')
infer=pd.read_csv(V/'stage5_kaggle/inference.csv')
final=pd.read_csv(V/'stage6_kaggle/final_mean_std.csv')
tables={};br=[]
for r in back.itertuples():br.append([r.exp_id,r.backbone.split('.')[0],r.backbone,r.params_m,r.gmacs,r.img_size,r.epochs_completed,r.seed,r.val_macro_f1,r.val_top1,r.train_val_seconds_per_epoch,r.latency_p50_ms,r.latency_p95_ms,'FP32 batch1 forward-only; best val epoch '+str(r.best_epoch),'validation/stage3_kaggle/runs/'+r.exp_id+'/seed0/'])
tables['Backbones']=section(['exp_id','Backbone','Pretrained tag','Params (M)','GMAC','Size (px)','Epochs','Seed','Macro-F1 val','Top-1 val','Train+val /epoch (s)','p50 batch1 (ms)','p95 batch1 (ms)','Notes','Source'],br,'Stage3: full fold0, shared recipe, seed0, Tesla T4. Runtime includes epoch validation; latency excludes softmax and preprocessing.',[24,23,43,13,12,12,10,8,16,15,22,20,20,56,70])
tr=[]
for r in training.itertuples():
 note='Seed0 screening; not a multi-seed claim.'
 if r.exp_id=='T07_combined':note='Same recipe/seed as T05; no multi-factor interaction was tested.'
 tr.append([r.exp_id,r.backbone,r.axis,r.changes_from_T00,r.seed,r.val_macro_f1,r.val_top1,r.delta_macro_f1_vs_T00,r.val_f1_class0,r.val_f1_class7,r.val_ece,r.best_epoch,note,'validation/stage4_kaggle/runs/'+r.exp_id+'/seed0/'])
for exp in ('S01_full_epoch','S02_resume'):
 d=V/f'stage2_kaggle/runs/{exp}/seed0';r=load(d/'summary.json');cfg=load(d/'config.json')
 tr.append([exp,cfg['backbone'],'diagnostic','Pipeline check, not a controlled T00 ablation',0,r['val_macro_f1'],r['val_top1'],None,None,None,None,r['best_epoch'],
            'S01: full val,1epoch only; S02:18train/9val debug subset. Neither used for selection.',str(d.relative_to(SUB)).replace('\\','/')])
tables['Training']=section(['exp_id','Backbone tag','Axis','Change from T00','Seed','Macro-F1 val','Top-1 val','Delta macro-F1','F1 Chinee apple','F1 Snake weed','ECE val','Best epoch','Notes','Source'],tr,'Stage4 fixed baseline: T00 CE +crop/flip. A initialization; B augmentation; C loss. Values are validation seed0, not test.',[25,40,13,30,9,16,15,18,19,19,16,13,66,68])
ir=[];lat=[]
for r in infer.itertuples():
 d=V/f'stage5_kaggle/methods/{r.training_exp}/{r.method}';a=load(d/'report.json')
 checkpoint=a['checkpoint_sha256']
 ir.append([r.training_exp+'_'+r.method,r.method,r.training_exp+' seed0',r.views,r.val_macro_f1,r.val_top1,r.val_ece,r.p50_ms,r.p95_ms,r.p99_ms,r.batch32_images_per_s,r.relative_cost,r.dtype,r.temperature,checkpoint,'validation/stage5_kaggle/methods/'+r.training_exp+'/'+r.method+'/'])
 for batch in (1,32):
  l=a[f'latency_batch{batch}'];lat.append([r.training_exp+'_'+r.method,'stage5',l['gpu'],l['dtype'],batch,l['resolution'],l['views'],'No',l['p50'],l['p95'],l['p99'],l['images_per_s'],10,l['n'],'view+forward+aggregation; input already on GPU',str(d.relative_to(SUB)).replace('\\','/')+'/report.json'])
tables['Inference']=section(['exp_id','Method','Training checkpoint','K views','Macro-F1 val','Top-1 val','ECE val','p50 batch1 (ms)','p95 batch1 (ms)','p99 batch1 (ms)','Batch32 images/s','Relative p50 cost','Dtype','Temperature','Checkpoint SHA256','Source'],ir,'Stage5: all3501 val images. Temperature is fit on full val224 and evaluated on the same set. Latency includes GPU view/aggregation, excludes image preprocessing and transfer.',[38,25,28,10,17,16,15,20,20,20,21,21,12,17,70,80])
fr=[];aggregate_info={};seed_metrics={};per=[]
for exp in ('F01','T00'):
 cfg='ConvNeXt-Tiny, LS0.1,288 FP32+val T' if exp=='F01' else 'ConvNeXt-Tiny, CE,224 FP32,T1'
 for seed in (0,1,2):
  d=V/f'stage6_kaggle/runs/{exp}/seed{seed}/final_evaluation';a=load(d/'val_report.json');b=load(d/'test_report.json')
  seed_metrics[(exp,seed)]={'val':a['metrics'],'test':b['metrics']}
  fr.append([exp,cfg,seed,a['metrics']['macro_f1'],b['metrics']['macro_f1'],b['metrics']['top1'],b['metrics']['ece'],None,None,None,a['latency_batch1']['p95'],a['temperature'],str(d.relative_to(SUB)).replace('\\','/')])
  for batch in (1,32):
   l=a[f'latency_batch{batch}'];lat.append([f'{exp}_seed{seed}','stage6',l['gpu'],l['dtype'],batch,l['resolution'],l['views'],'No',l['p50'],l['p95'],l['p99'],l['images_per_s'],10,l['n'],'view+forward+temperature; preprocessing/transfer excluded',str(d.relative_to(SUB)).replace('\\','/')+'/val_report.json'])
for exp in ('F01','T00'):
 v=final[(final.exp_id==exp)&(final.split=='val')].iloc[0];t=final[(final.exp_id==exp)&(final.split=='test')].iloc[0]
 p95=max(row[10] for row in fr if row[0]==exp)
 fr.append([exp,'Aggregate3 seeds; std sample ddof1','aggregate 3 seeds',v.macro_f1_mean,t.macro_f1_mean,t.top1_mean,t.ece_mean,t.macro_f1_std,t.top1_std,t.ece_std,p95,None,'official/'+exp+'_summary.json'])
 aggregate_info[exp]={'val':v.to_dict(),'test':t.to_dict(),'p95_ms_max':p95}
tables['Final']=section(['exp_id','Frozen configuration','Seed','Macro-F1 val','Macro-F1 test','Top-1 test','ECE test','Macro-F1 std','Top-1 std','ECE std','p95 batch1 (ms)','Temperature','Source'],fr,'Stage6: fixed before test. F01 val/test at288; train/checkpoint selection at224. Calibration fits val per seed. T00 is the uncalibrated224 baseline.',[14,55,20,18,18,18,16,17,16,16,22,18,90])
counts=pd.read_csv(SUB/'labels/test_subset0.csv').Label.value_counts()
for exp in ('F01','T00'):
 a=load(SUB/f'official/{exp}_summary.json')
 for k,name in enumerate(CLASS_NAMES):per.append([exp,name,int(counts[k]),a['precision']['mean'][k],a['precision']['std'][k],a['recall']['mean'][k],a['recall']['std'][k],a['f1']['mean'][k],a['f1']['std'][k],'mean3 seeds; ddof1',f'official/{exp}_per_class.csv'])
tables['PerClass']=section(['exp_id','Class','Test images /seed','Precision mean','Precision std','Recall mean','Recall std','F1 mean','F1 std','Aggregation','Source'],per,'Per-class values are mean/std across3 independent seeds, not metrics from a pooled confusion matrix. Class IDs follow original Label.',[14,23,22,19,18,19,18,18,18,33,45])
tables['Latency']=section(['Configuration','Stage','GPU','Dtype','Batch','Resolution (px)','K views','BN fused','p50 (ms)','p95 (ms)','p99 (ms)','Images/s','Warmup','Measurements','Scope','Source'],lat,'Measurements: synchronize before/after each iteration,10warmup100 measured; mean/percentiles/raw samples preserved in source JSON. Stage3 is forward-only and is listed on Backbones instead.',[40,13,17,11,10,20,12,13,17,17,17,19,13,18,70,90])
# Summary top10 across all completed val configurations. Rows label stage/seed count to avoid false comparability.
candidates=[]
for r in back.itertuples():candidates.append([r.exp_id,'Backbone,1 seed',r.val_macro_f1,None,r.latency_p95_ms,1,r.val_top1,'FP32 forward-only'])
for r in training.itertuples():candidates.append([r.exp_id,'Training,1 seed',r.val_macro_f1,None,None,1,r.val_top1,'Latency not measured here'])
for r in infer.itertuples():candidates.append([r.training_exp+'_'+r.method,'Inference,1 seed',r.val_macro_f1,None,r.p95_ms,1,r.val_top1,r.dtype])
for exp in ('F01','T00'):
 a=aggregate_info[exp];candidates.append([exp+' final','Final,3 seeds',a['val']['macro_f1_mean'],a['val']['macro_f1_std'],a['p95_ms_max'],3,a['val']['top1_mean'],'p95=max across seeds'])
candidates.sort(key=lambda r:(-r[2],r[4] if r[4] is not None else 1e9,r[0]))
tables['Summary']=section(['Configuration','Experiment scope','Macro-F1 val','Val std','p95 batch1 (ms)','Seeds','Top-1 val','Latency condition'],candidates[:10],'Ranking compares validation candidates from screening and final; different seed counts/scopes are explicit. Final test results above are not used for ranking or selection.',[41,24,19,16,24,10,18,35])
# Mistake examples chosen AFTER freeze/test for explanation only; no model is run.
p=read_pred(str(SUB/'predictions/F01_seed0_test.csv'));wrong=np.flatnonzero(p.y_true!=p.y_pred)
err=pd.DataFrame({'Filename':p.filenames[wrong],'y_true':p.y_true[wrong],'y_pred':p.y_pred[wrong],'confidence':p.probs[wrong].max(1)})
err.to_csv(figs/'F01_seed0_errors.csv',index=False)
preferred=err[(err.y_true.isin([0,7]))|(err.y_pred.isin([0,7]))].sort_values('confidence',ascending=False)
other=err.sort_values('confidence',ascending=False)
selected=pd.concat([preferred.head(6),other]).drop_duplicates('Filename').head(9)
selected.to_csv(figs/'error_sample_manifest.csv',index=False)
with zipfile.ZipFile(ROOT/'images.zip') as z:
 index={Path(n).name:n for n in z.namelist() if n.lower().endswith('.jpg')}
 fig,axes=plt.subplots(3,3,figsize=(12,13))
 for ax,r in zip(axes.flat,selected.itertuples()):
  im=Image.open(io.BytesIO(z.read(index[r.Filename]))).convert('RGB');ax.imshow(im);ax.axis('off')
  ax.set_title(f'{r.Filename}\nTrue: {CLASS_NAMES[r.y_true]}\nPred: {CLASS_NAMES[r.y_pred]}, p={r.confidence:.3f}',fontsize=9)
 fig.suptitle('F01 seed0: test mistakes inspected after final recipe was frozen',fontsize=14)
 fig.tight_layout();fig.savefig(figs/'error_samples.png',dpi=150);plt.close(fig)
cm=np.asarray(seed_metrics[('F01',0)]['test']['confusion'])
confusions=[]
for i in range(9):
 for j in range(9):
  if i!=j and cm[i,j]:confusions.append({'true':CLASS_NAMES[i],'predicted':CLASS_NAMES[j],'count_seed0':int(cm[i,j])})
confusions.sort(key=lambda x:-x['count_seed0'])
(figs/'top_confusions.json').write_text(json.dumps(confusions,indent=2),encoding='utf-8')
# Reliability chart from saved test probabilities; diagnosis, not fitting.
fig,ax=plt.subplots(figsize=(7,6))
for name,label in [('F01_uncal','Before temperature'),('F01','After val-fitted temperature')]:
 q=read_pred(str(SUB/f'predictions/{name}_seed0_test.csv'));confidence=q.probs.max(1);correct=(q.y_pred==q.y_true)
 xs=[];ys=[]
 for j in range(15):
  mask=(confidence>j/15)&(confidence<=(j+1)/15)
  if mask.any():xs.append(float(confidence[mask].mean()));ys.append(float(correct[mask].mean()))
 ax.plot(xs,ys,'o-',label=label)
ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='Mean confidence per bin',ylabel='Accuracy per bin',title='F01 seed0 test calibration; T fitted on val');ax.legend();fig.tight_layout();fig.savefig(figs/'reliability_seed0.png',dpi=150);plt.close(fig)
for src,dest in [(V/'stage3_kaggle/backbone_tradeoff.png',figs/'backbone_tradeoff.png'),(V/'stage4_kaggle/training_ablation.png',figs/'training_ablation.png'),(V/'stage5_kaggle/inference_tradeoff.png',figs/'inference_tradeoff.png')]:shutil.copyfile(src,dest)
# Manifest links every curve to its actual run/epoch log.
curve_rows=[]
for stage in (2,3,4,6):
 for path in (V/f'stage{stage}_kaggle/runs').glob('*/seed*/history.csv'):
  exp,seed=path.parent.parent.name,path.parent.name
  curve_name=f'{exp}_{seed}.png' if not(stage==4 and exp=='T00') else f'T00_stage4_{seed}.png'
  curve_rows.append({'stage':stage,'exp_id':exp,'seed':int(seed[4:]),'curve':'curves/'+curve_name,'history':str(path.relative_to(SUB)).replace('\\','/'),'config':str((path.parent/'config.json').relative_to(SUB)).replace('\\','/')})
for stage in (2,):
 for source in (V/f'stage{stage}_kaggle/runs').glob('*/seed*/curves.png'):
  shutil.copyfile(source,SUB/'curves'/f'{source.parent.parent.name}_{source.parent.name}.png')
shutil.copyfile(V/'stage2_kaggle/smoke/overfit_curve.png',SUB/'curves/S00_overfit9.png')
pd.DataFrame(curve_rows).to_csv(SUB/'curves/manifest.csv',index=False)
# No non-finite placeholders masquerading as zero in workbook.
payload={'tables':tables,'headline':aggregate_info,'top_confusions':confusions,'selected_errors':selected.to_dict('records')}
work=ROOT/'tools/stage7_workbook';work.mkdir(exist_ok=True)
(work/'results_data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
(SUB/'validation/stage7_audit/results_data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print('Prepared7 sheets;',len(curve_rows),'curves;',len(err),'seed0 mistakes;',len(selected),'image examples.')
print('Top confusions:',confusions[:7])
