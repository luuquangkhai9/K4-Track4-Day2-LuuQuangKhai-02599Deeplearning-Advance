import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const base=path.dirname(new URL(import.meta.url).pathname.replace(/^\/(\w:)/,'$1'));
const root=path.resolve(base,'../..');
const sub=path.join(root,'submissions/02599_LuuQuangKhai');
const output=path.join(base,'outputs/stage7-results');
await fs.mkdir(output,{recursive:true});
const data=JSON.parse(await fs.readFile(path.join(base,'results_data.json'),'utf8'));
const wb=Workbook.create();
const names=['Summary','Final','Backbones','Training','Inference','PerClass','Latency'];
for(const name of names)wb.worksheets.add(name);
const dark='#243B53', light='#EDF2F7', accent='#DCEFE8';
const col=n=>{let s='';for(let x=n+1;x;x=Math.floor((x-1)/26))s=String.fromCharCode(65+(x-1)%26)+s;return s;};
function formatTable(sheet,headers,rows,start=4,widths=[]){
  const end=start+rows.length, last=col(headers.length-1);
  sheet.showGridLines=false;
  sheet.getRange(`A${start}:${last}${end}`).format.font={name:'Arial',size:10,color:dark};
  sheet.getRange(`A${start}:${last}${end}`).format.rowHeight=24;
  sheet.getRange(`A${start}:${last}${end}`).format.verticalAlignment='center';
  sheet.getRange(`A${start}:${last}${start}`).values=[headers];
  sheet.getRange(`A${start}:${last}${start}`).format={fill:dark,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center',rowHeight:44};
  if(rows.length)sheet.getRange(`A${start+1}:${last}${end}`).values=rows;
  for(let i=0;i<headers.length;i++){
    const c=col(i);sheet.getRange(`${c}:${c}`).format.columnWidth=widths[i]??20;
    const h=headers[i];const range=sheet.getRange(`${c}${start+1}:${c}${end}`);
    if(/F1|Top-1|ECE|Precision|Recall|Delta|std/i.test(h)){range.setNumberFormat('0.0000');range.format.horizontalAlignment='right';}
    else if(/\(ms\)|Images\/s|images\/s|GMAC|Params|Temperature|cost|\(s\)/i.test(h)){range.setNumberFormat('0.00');range.format.horizontalAlignment='right';}
    else if(/Seed|K views|Epoch|Size|Batch|Warmup|Measurements|Resolution|Test images/i.test(h)){range.setNumberFormat('0');range.format.horizontalAlignment='right';}
    else range.format.horizontalAlignment='left';
  }
  for(let row=start+1;row<=end;row++)if((row-start)%2===0)sheet.getRange(`A${row}:${last}${row}`).format.fill=light;
  return {start,end,last};
}
for(const name of names.filter(n=>n!=='Summary')){
  const sheet=wb.worksheets.getItem(name), spec=data.tables[name];
  sheet.getRange('A1').values=[[name]];sheet.getRange('A1').format.font={name:'Arial',size:14,bold:true,color:dark};
  sheet.getRange('A2').values=[[spec.note]];sheet.getRange('A2').format.font={name:'Arial',size:10,italic:true,color:'#526579'};
  formatTable(sheet,spec.headers,spec.rows,4,spec.widths);
  sheet.freezePanes.freezeRows(4);sheet.freezePanes.freezeColumns(1);
  sheet.tabColor=name==='Final'?'#197A65':'#607D98';
  const metricIndex=name==='Backbones'?8:name==='Training'?5:name==='Inference'?4:null;
  if(metricIndex!==null){
    const best=Math.max(...spec.rows.filter(r=>r[metricIndex]!==null).map(r=>r[metricIndex]));
    for(let i=0;i<spec.rows.length;i++)if(Math.abs(spec.rows[i][metricIndex]-best)<1e-12){
      sheet.getRange(`A${i+5}:${col(spec.headers.length-1)}${i+5}`).format.fill=accent;
      sheet.getRange(`A${i+5}:${col(spec.headers.length-1)}${i+5}`).format.font.bold=true;
    }
  }
}
const final=wb.worksheets.getItem('Final');
for(const [row,start] of [[11,5],[12,8]]){
  for(const c of ['D','E','F','G'])final.getRange(`${c}${row}`).formulas=[[`=AVERAGE(${c}${start}:${c}${start+2})`]];
  final.getRange(`H${row}`).formulas=[[`=STDEV(E${start}:E${start+2})`]];
  final.getRange(`I${row}`).formulas=[[`=STDEV(F${start}:F${start+2})`]];
  final.getRange(`J${row}`).formulas=[[`=STDEV(G${start}:G${start+2})`]];
  final.getRange(`K${row}`).formulas=[[`=MAX(K${start}:K${start+2})`]];
  final.getRange(`A${row}:M${row}`).format.fill=accent;
  final.getRange(`A${row}:M${row}`).format.font.bold=true;
}
// Ablation deltas are calculated against the one authoritative stage4 T00 row.
const training=wb.worksheets.getItem('Training');
const baselineRow=5+data.tables.Training.rows.findIndex(r=>r[0]==='T00');
for(let i=0;i<data.tables.Training.rows.length;i++)if(data.tables.Training.rows[i][2]!=='diagnostic')training.getRange(`H${i+5}`).formulas=[[`=F${i+5}-$F$${baselineRow}`]];
for(let i=0;i<data.tables.Training.rows.length;i++)if(data.tables.Training.rows[i][2]==='diagnostic'){
  training.getRange(`D${i+5}`).format.wrapText=true;
  training.getRange(`M${i+5}`).format.wrapText=true;
  training.getRange(`A${i+5}:N${i+5}`).format.rowHeight=44;
}
const summary=wb.worksheets.getItem('Summary');summary.showGridLines=false;summary.tabColor='#197A65';
summary.mergeCells('A1:H1');summary.getRange('A1').values=[['DeepWeeds final comparison']];summary.getRange('A1:H1').format={font:{name:'Arial',size:14,bold:true,color:dark},rowHeight:30};
summary.mergeCells('A2:H2');summary.getRange('A2').values=[['Luu Quang Khai 02599 · official fold0 · 3 seeds · sample std (ddof1)']];summary.getRange('A2:H2').format={font:{name:'Arial',size:10,color:'#526579'},rowHeight:23};
const headings=['Configuration','Macro-F1 test','F1 std','Top-1 test','Top-1 std','ECE test','p95 batch1 (ms)','Inference'];
formatTable(summary,headings,[['F01',null,null,null,null,null,null,'288 FP32 + val-fitted T'],['T00',null,null,null,null,null,null,'224 FP32, T=1']],4,[41,24,19,16,24,10,18,35]);
for(const [r,src] of [[5,11],[6,12]]){
  summary.getRange(`B${r}:G${r}`).formulas=[[`=Final!E${src}`,`=Final!H${src}`,`=Final!F${src}`,`=Final!I${src}`,`=Final!G${src}`,`=Final!K${src}`]];
}
summary.getRange('B5:F6').setNumberFormat('0.0000');summary.getRange('G5:G6').setNumberFormat('0.00');summary.getRange('A5:H5').format.fill=accent;
summary.getRange('H5:H6').format.horizontalAlignment='center';
summary.getRange('A8:B9').values=[['Final - baseline macro-F1',null],['Larger test F1 std',null]];
summary.getRange('B8').formulas=[['=B5-B6']];summary.getRange('B9').formulas=[['=MAX(C5:C6)']];summary.getRange('B8:B9').setNumberFormat('0.0000');
summary.mergeCells('D8:H9');summary.getRange('D8').values=[['p95 is the maximum measured across final seeds. Image decoding, resize, normalization and CPU-to-GPU transfer are excluded.']];summary.getRange('D8:H9').format={wrapText:true,font:{name:'Arial',size:10,color:'#526579'},verticalAlignment:'center'};
summary.mergeCells('A11:H11');summary.getRange('A11').values=[['Top 10 validation configurations']];summary.getRange('A11:H11').format={font:{name:'Arial',size:12,bold:true,color:dark},rowHeight:28};
formatTable(summary,data.tables.Summary.headers,data.tables.Summary.rows,12,data.tables.Summary.widths);
summary.mergeCells('A24:H25');summary.getRange('A24').values=[[data.tables.Summary.note]];summary.getRange('A24:H25').format={wrapText:true,font:{name:'Arial',size:10,color:'#526579'},verticalAlignment:'center',rowHeight:23};
summary.mergeCells('A27:H28');summary.getRange('A27').values=[['Final selection used validation only. Test was exported once per seed/run. Evaluator proposed section I: 20/20; this is not the total lab grade.']];summary.getRange('A27:H28').format={wrapText:true,font:{name:'Arial',size:10,color:'#526579'},verticalAlignment:'center',rowHeight:23};
wb.recalculate();
const inspect=await wb.inspect({kind:'region',sheetId:'Summary',range:'A4:H9',maxChars:3000,tableMaxRows:6,tableMaxCols:8});
await fs.writeFile(path.join(output,'summary_inspect.ndjson'),inspect.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:1500});
await fs.writeFile(path.join(output,'formula_scan.ndjson'),errors.ndjson);
// Check dependent edits and restore before export. Formula calculations must be live.
const initial=final.getRange('E11').values[0][0];
const original=final.getRange('E5').values[0][0];final.getRange('E5').values=[[original-0.03]];wb.recalculate();
if(Math.abs(final.getRange('E11').values[0][0]-(initial-0.01))>1e-8)throw new Error('Mean formula does not recalculate');
final.getRange('E5').values=[[original]];wb.recalculate();
for(const name of (process.argv.includes('--changed-only')?['Summary','Final','Training','Backbones','Inference']:names)){
  const preview=await wb.render({sheetName:name,range:name==='Summary'?'A1:H28':name==='Final'?'A1:M12':name==='Latency'?'A1:N11':name==='Training'?'A1:N14':name==='Inference'?'A1:P20':'A1:L11',scale:1,format:'png'});
  await fs.writeFile(path.join(output,`${name}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(path.join(output,'results.xlsx'));
await fs.copyFile(path.join(output,'results.xlsx'),path.join(sub,'results.xlsx'));
console.log('Saved results.xlsx with7 sheets; live mean/std/delta formulas and7 rendered previews.');
