#!/usr/bin/env python3
"""Cross-layer completion report, paired statistics, PDF figures and HTML atlas."""
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

MASKS=['temporal_front','temporal_middle','temporal_back','spatial_block','spatial_scattered']
LABELS=['隐藏第1–2帧','隐藏第7–8帧','隐藏第15–16帧','连续空间遮挡','分散空间遮挡']
TARGETS=['l1','l2','cosine','inner_product']


def writecsv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def savejson(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def interval(x):
    x=np.asarray(x,float);rng=np.random.default_rng(240924)
    draws=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975)),n=len(x))


def setup_style():
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family=font_manager.FontProperties(fname=str(font)).get_name()
    else:family='DejaVu Sans'
    plt.rcParams.update({'font.family':family,'font.size':10,'axes.unicode_minus':False,
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})


def savefig(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)


def image_panel(rgb,hidden,heat=None):
    image=rgb.astype(float)/255
    if heat is not None:
        h=heat.repeat(16,0).repeat(16,1)
        color=plt.get_cmap('inferno')(h)[...,:3]
        image=image*(1-.82*h[...,None])+color*.82*h[...,None]
    hm=hidden.repeat(16,0).repeat(16,1)
    yy,xx=np.indices((256,256));checker=np.where(((yy//8+xx//8)%2)[...,None],.32,.40)
    return np.uint8(np.clip(np.where(hm[...,None],checker,image)*255,0,255))


def overview(out,stem,rgb,visible,hidden,heat):
    full=np.zeros((12,2048));full[:,visible]=heat[:,0]
    scaled=full/np.maximum(full.max(1,keepdims=True),1e-30)
    scaled=scaled.reshape(12,8,16,16);hm=np.isin(np.arange(2048),hidden).reshape(8,16,16)
    for start in (0,6):
        fig,axes=plt.subplots(7,8,figsize=(16,14))
        fig.subplots_adjust(wspace=.025,hspace=.10,left=.06,right=.99,top=.96,bottom=.09)
        for t in range(8):
            axes[0,t].imshow(image_panel(rgb[2*t+1],hm[t]));axes[0,t].set_title(f'{2*t+1}–{2*t+2}帧')
            for offset in range(6):axes[offset+1,t].imshow(image_panel(rgb[2*t+1],hm[t],scaled[start+offset,t]))
        for ax in axes.ravel():ax.set_xticks([]);ax.set_yticks([])
        axes[0,0].set_ylabel('可见输入')
        for offset in range(6):axes[offset+1,0].set_ylabel(f'P{start+offset+1}')
        cax=fig.add_axes([.36,.063,.33,.009])
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,1),cmap='inferno'),cax=cax,orientation='horizontal',ticks=[0,.5,1])
        cb.ax.tick_params(labelsize=8)
        fig.text(.5,.018,stem+'\n同一最终负L1目标；每层独立归一化，层内8个时间片共用尺度。\n'
            '灰格是隐藏位置；热图表示该层可见token位置的注意力分支归因，亮度不代表层重要性。',ha='center')
        savefig(fig,out/'figures'/f'{stem}_P{start+1}-{start+6}')
    # A compact 12-layer view at a deterministic nearest visible time bin.
    hidden_times=np.unique(hidden//256)
    t=6 if np.array_equal(hidden_times,np.array([7])) else 1 if np.array_equal(hidden_times,np.array([0])) else 2
    fig,axes=plt.subplots(3,4,figsize=(10,8.6))
    for l,ax in enumerate(axes.ravel()):
        ax.imshow(image_panel(rgb[2*t+1],hm[t],scaled[l,t]));ax.set_xlabel(f'P{l+1}')
        ax.set_xticks([]);ax.set_yticks([])
    fig.subplots_adjust(left=.02,right=.99,bottom=.16,top=.99,wspace=.07,hspace=.24)
    cax=fig.add_axes([.30,.08,.40,.012])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,1),cmap='inferno'),cax=cax,orientation='horizontal',ticks=[0,.5,1])
    cb.ax.tick_params(labelsize=8)
    fig.text(.5,.011,stem+f' · 第{2*t+1}–{2*t+2}帧 · 最终负L1目标\n色标：正归因 / 该层完整时间范围的最大值。层间只比较位置分布。',ha='center',fontsize=9)
    savefig(fig,out/'figures'/f'{stem}_12layers')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('/mnt/sdb/lanxin/data/latentflow/kubric_movi_c_l08'))
    a=p.parse_args();out=a.root/'report';out.mkdir(exist_ok=True)
    for name in ['figures','inputs']:(out/name).mkdir(exist_ok=True)
    shutil.copy2(__file__,out/Path(__file__).name);setup_style()
    runs=sorted(a.root.glob('test_shard*'));assert runs and all((r/'complete.json').exists() for r in runs)
    cases={};curves=[]
    for run in runs:
        for f in sorted((run/'cases').glob('*.json')):
            assert f.stem not in cases;cases[f.stem]=(run,json.loads(f.read_text()))
        for f in sorted((run/'curves').glob('*.csv')):
            with f.open() as stream:curves.extend(csv.DictReader(stream))
    assert len(cases)==70 and len(curves)==7140
    scenes=sorted({c['scene'] for _,c in cases.values()});assert len(scenes)==14
    temporal=[];similarities=[];diagnostics=[];payload={};bycase={}
    example=scenes[0]
    rgb_cache={}
    for stem,(run,case) in sorted(cases.items()):
        sid=case['scene'];mask=case['mask']
        if sid not in rgb_cache:rgb_cache[sid]=np.load(a.dataset/sid/'rgb.npy')
        rgb=rgb_cache[sid]
        with np.load(run/'maps'/f'{stem}.npz') as z:
            heat=z['heat'].copy();hidden_heat=z['hidden_query_heat'].copy();visible=z['visible'];hidden=z['hidden']
            full=np.zeros((12,4,2048));full[:,:,visible]=heat
            hid=np.isin(np.arange(2048),hidden).reshape(8,16,16)
            normalized=full/np.maximum(full.max(2,keepdims=True),1e-30)
            payload[stem]=dict(values=np.rint(normalized*255).astype(np.uint8).reshape(12,4,8,256).tolist(),
                hidden=np.isin(np.arange(2048),hidden).astype(int).reshape(8,256).tolist(),
                zero=(heat.max(2)==0).tolist(),baseline=case['original_scores'])
            for l in range(12):
                for ti,target in enumerate(TARGETS):
                    total=float(heat[l,ti].sum())
                    diagnostics.append(dict(scene=sid,mask=mask,layer=l+1,target=target,positive_mass=total,
                        positive_count=int(np.sum(heat[l,ti]>0)),
                        zero=total==0,hidden_query_cosine=(float(np.dot(heat[l,ti],hidden_heat[l,ti])/
                        (np.linalg.norm(heat[l,ti])*np.linalg.norm(hidden_heat[l,ti])))
                        if total>0 and hidden_heat[l,ti].sum()>0 else None)))
                    for t in range(8):
                        temporal.append(dict(scene=sid,mask=mask,layer=l+1,target=target,time=t,
                            fraction=float(full[l,ti,t*256:(t+1)*256].sum()/total) if total>0 else None,
                            uniform_fraction=float(np.mean(visible//256==t))))
                for other in range(12):
                    valid=bool(heat[l,0].sum()>0 and heat[other,0].sum()>0)
                    r1=z[f'rank_L{l+1:02d}'][:179];r2=z[f'rank_L{other+1:02d}'][:179]
                    similarities.append(dict(scene=sid,mask=mask,layer=l+1,other=other+1,
                        cosine=float(np.dot(heat[l,0],heat[other,0])/(np.linalg.norm(heat[l,0])*np.linalg.norm(heat[other,0]))) if valid else None,
                        top179_overlap=len(set(r1)&set(r2))/179 if valid and min(np.sum(heat[l,0]>0),np.sum(heat[other,0]>0))>=179 else None))
        frames=np.concatenate([image_panel(rgb[2*t+1],hid[t]) for t in range(8)],axis=1)
        Image.fromarray(frames).save(out/'inputs'/f'{stem}.jpg',quality=88)
        if sid==example and mask in ('temporal_front','temporal_middle','temporal_back'):
            overview(out,stem,rgb,visible,hidden,heat)
    for r in curves:bycase[(r['scene'],r['mask'],r['fill'],r['method'])]=r
    gains=[]
    for sid in scenes:
        for mask in MASKS:
            for fill in ('mean','blur'):
                for l in range(1,13):
                    name=f'L{l:02d}';drop=float(bycase[(sid,mask,fill,name)]['l1_drop'])
                    base=float(bycase[(sid,mask,fill,'L12')]['l1_drop'])
                    tm=np.mean([float(bycase[(sid,mask,fill,f'{name}_time_{s}')]['l1_drop']) for s in (41,42,43)])
                    rr=np.mean([float(bycase[(sid,mask,fill,f'random_{s}')]['l1_drop']) for s in (41,42,43)])
                    gains.append(dict(scene=sid,mask=mask,fill=fill,layer=l,drop=drop,
                        time_gain=drop-tm,random_gain=drop-rr,vs_p12=drop-base))
    summary=[]
    for mask in MASKS+['all']:
        for fill in ('mean','blur'):
            for l in range(1,13):
                selected=[r for r in gains if (r['mask']==mask or mask=='all') and r['fill']==fill and r['layer']==l]
                for metric in ('drop','time_gain','random_gain','vs_p12'):
                    values=[np.mean([r[metric] for r in selected if r['scene']==sid]) for sid in scenes]
                    summary.append(dict(mask=mask,fill=fill,layer=l,metric=metric,**interval(values)))
    for name,rows in [('temporal_mass',temporal),('layer_similarity',similarities),('map_diagnostics',diagnostics),
                       ('deletion_per_scene',gains),('deletion_summary',summary)]:writecsv(out/(name+'.csv'),rows)
    valid_counts=[]
    for mask in MASKS:
        for l in range(1,13):
            for other in range(1,13):
                rows=[r for r in similarities if r['mask']==mask and r['layer']==l and r['other']==other]
                valid_counts.append(dict(mask=mask,layer=l,other=other,
                    cosine_scenes=sum(r['cosine'] is not None for r in rows),
                    top179_scenes=sum(r['top179_overlap'] is not None for r in rows)))
    writecsv(out/'similarity_valid_counts.csv',valid_counts)
    # Paired scene intervals: the same resampling seed preserves pairing across layers.
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.7),sharex=True)
    for ax,mask,label in zip(axes.ravel(),['all']+MASKS,['五种遮挡等权平均']+LABELS):
        ax.axhline(0,color='.5',lw=.8)
        for fill,color,marker in [('mean','#0072B2','o'),('blur','#D55E00','s')]:
            rows=[r for r in summary if r['mask']==mask and r['fill']==fill and r['metric']=='time_gain']
            y=np.array([r['mean'] for r in rows]);lo=np.array([r['ci_low'] for r in rows]);hi=np.array([r['ci_high'] for r in rows])
            ax.errorbar(range(1,13),y,yerr=[y-lo,hi-y],color=color,marker=marker,ms=3,lw=1,label=fill,capsize=2)
        ax.set_xlabel('Predictor层');ax.set_ylabel('相对时间匹配随机的L1误差增量')
        ax.text(.03,.97,label,transform=ax.transAxes,va='top',fontsize=10)
        ax.set_xticks([1,3,6,9,12])
    axes[0,0].legend(loc='lower left',fontsize=9)
    fig.tight_layout();savefig(fig,out/'figures'/'deletion_by_layer')
    fig,axes=plt.subplots(1,5,figsize=(15.2,5.1))
    for ax,mask,label in zip(axes,MASKS,LABELS):
        matrix=np.full((12,8),np.nan)
        for l in range(1,13):
            for t in range(8):
                vals=[r['fraction'] for r in temporal if r['mask']==mask and r['target']=='l1' and r['layer']==l and r['time']==t and r['fraction'] is not None and r['uniform_fraction']>0]
                if vals:matrix[l-1,t]=np.mean(vals)
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        ax.set_xticks(range(8));ax.set_xticklabels([f'{2*t+1}–{2*t+2}' for t in range(8)],rotation=90)
        ax.set_yticks(range(12));ax.set_yticklabels(range(1,13));ax.set_xlabel(label);ax.set_ylabel('Predictor层')
    fig.subplots_adjust(bottom=.23,right=.91,wspace=.45)
    cax=fig.add_axes([.925,.23,.012,.65]);fig.colorbar(im,cax=cax,label='正归因质量占比')
    fig.text(.5,.025,'固定最终负L1目标；14个场景平均。时间mask每个可见时间片的均匀基线为1/7，空间mask为1/8。\n白色为隐藏时间片；层内占比衡量分布，不是因果贡献百分比。',ha='center',fontsize=9)
    savefig(fig,out/'figures'/'temporal_by_layer')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,metric,label in zip(axes,['cosine','top179_overlap'],['归一化正图余弦相似度','前179个位置的重叠比例']):
        matrix=np.array([[np.mean([r[metric] for r in similarities if r['layer']==l and r['other']==other and r[metric] is not None]) for other in range(1,13)] for l in range(1,13)])
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='cividis');ax.set_xlabel('Predictor层');ax.set_ylabel('Predictor层')
        ax.set_xticks([0,2,5,8,11]);ax.set_xticklabels([1,3,6,9,12]);ax.set_yticks([0,2,5,8,11]);ax.set_yticklabels([1,3,6,9,12])
        fig.colorbar(im,ax=ax,label=label,shrink=.8)
    fig.tight_layout();savefig(fig,out/'figures'/'layer_similarity')
    colors=np.rint(plt.get_cmap('inferno')(np.arange(256)/255)[:,:3]*255).astype(int).tolist()
    (out/'data.js').write_text('window.ATLAS='+json.dumps(dict(scenes=scenes,masks=MASKS,labels=LABELS,targets=TARGETS,
        colors=colors,cases=payload),ensure_ascii=False,separators=(',',':'))+';\n')
    (out/'index.html').write_text(HTML)
    smoke=json.loads(next((a.root/'smoke_batch2'/'cases').glob('*.json')).read_text())
    audit=dict(scenes=14,cases=70,main_maps=len(diagnostics),deletion_points=len(curves),
        zero_maps=sum(r['zero'] for r in diagnostics),max_old_p12_error=max(c['old_p12_max_error'] for _,c in cases.values()),
        max_hook_error=max(c['hook_error'] for _,c in cases.values()),max_hidden_error=max(c['hidden_pixel_error'] for _,c in cases.values()),
        smoke_fd=smoke['finite_differences'],smoke_batch_single_score_error=smoke['batch_single_score_error'],
        sparse_maps=sum(r['positive_count']<179 for r in diagnostics),
        definition='all-query attention-branch attribution to visible token slots')
    savejson(out/'audit_summary.json',audit)
    report=['# 最终补全目标的predictor跨层归因','',
        '14个历史场景 × 5种mask × 12层 × 4种评分；固定最终预测与teacher目标，模型冻结。',
        '中间层汇总全部有梯度query，保留hidden/context query signed分量；P12复现旧实验。',
        '本轮归因位置为各层上下文化可见token槽位，不能等同信息最初来自同位置输入像素。',
        '逐层输入删除是选点质量验证，不是逐层计算重要性干预。每层独立热图归一化，不能比较亮度。','',
        '## 约10%输入删除：五mask先在scene内等权平均','',
        '|层|mean：L1增量|mean：相对时间匹配随机 [95% CI]|blur：相对时间匹配随机 [95% CI]|',
        '|---|---:|---|---|']
    for l in range(1,13):
        def get(fill,metric):return next(r for r in summary if r['mask']=='all' and r['fill']==fill and r['layer']==l and r['metric']==metric)
        x,y=get('mean','time_gain'),get('blur','time_gain')
        report.append(f"|P{l}|{get('mean','drop')['mean']:.6f}|{x['mean']:+.6f} [{x['ci_low']:+.6f}, {x['ci_high']:+.6f}]|{y['mean']:+.6f} [{y['ci_low']:+.6f}, {y['ci_high']:+.6f}]|")
    report += ['', '三次随机先在scene内平均；10000次14-scene配对bootstrap；点态区间未作多重比较校正。',
        '每层删除179/1792个可见tubelet（约10%）；四评分图均保存，删除排名以L1为主。',
        '所有层原始L1基线相同；没有每层重新预测或训练probe。零正图保留，形状相似度排除并报告计数。',
        '历史test重复使用，层排名只能作探索发现；从12层中事后挑最高值不能称确认性最佳层。','',
        f"零正图 {audit['zero_maps']}/{audit['main_maps']}；P12旧图最大差 {audit['max_old_p12_error']:.3g}。",
        '完整审核见audit_summary.json；独立重加载删除复算见../fresh_verification.json。','',
        '[交互图册](index.html) · [删除曲线](figures/deletion_by_layer.png) · [时间分布](figures/temporal_by_layer.png) · [层间相似度](figures/layer_similarity.png)']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    (out/'latex_includes.tex').write_text('\n'.join('\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{'+name+'.pdf}\n\\caption{'+caption+'}\n\\end{figure}\n' for name,caption in [
        ('figures/deletion_by_layer','Fixed-output completion attribution across predictor layers. Extra L1 error after deleting the top 179 visible tubelets relative to time-matched random locations; 95 percent paired scene bootstrap intervals.'),
        ('figures/temporal_by_layer','Temporal distribution of positive attribution across predictor layers. Each layer uses the same final negative L1 objective; mass fractions are descriptive, not causal contribution fractions.'),
        ('figures/layer_similarity','Layerwise similarity of visible-token attribution maps for the fixed final negative L1 score. Zero maps are excluded and counted in the audit report.')]))
    savejson(out/'complete.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='smoke_fd'}))


HTML='''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Completion Attribution Across Native Layers</title>
<style>
:root{color-scheme:light;--ink:#1f2933;--muted:#52606d;--line:#d9e2ec;--accent:#2563eb}
*{box-sizing:border-box}body{max-width:1660px;margin:28px auto;padding:0 22px;font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink)}
h1{font-size:28px;line-height:1.2;margin:0 0 8px}p{max-width:1120px}.lede{font-size:16px;margin:0 0 16px;color:#364152}.controls{position:sticky;top:0;background:#fff;padding:12px 0;border-bottom:1px solid var(--line);z-index:2;display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px}.controls label{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}.controls select,.controls button{font:inherit;padding:6px 9px;border:1px solid #bcccdc;border-radius:5px;background:#fff;color:inherit}.controls button{cursor:pointer;color:#fff;background:var(--accent);border-color:var(--accent)}.controls button:hover{background:#1d4ed8}.note{color:var(--muted);font-size:14px}.status{font-variant-numeric:tabular-nums;color:#334e68;margin:14px 0 4px}canvas{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fff}.preview{margin-top:24px}.preview[hidden]{display:none}.preview img{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fff}
</style>
</head>
<body>
<h1>How one final completion target is attributed across native layers</h1>
<p class="lede">Frozen V-JEPA is evaluated with one fixed target: the negative L1 score on the hidden target tokens. Switch between the 12 predictor blocks and the 24 encoder blocks to inspect where the same score receives positive attribution.</p>
<div class="controls">
  <label>Scene <select id="scene"></select></label>
  <label>Mask <select id="mask"></select></label>
  <label>Branch <select id="branch"><option value="predictor">Predictor (12 blocks)</option><option value="encoder">Encoder (24 blocks)</option></select></label>
  <label>View <select id="view"><option value="all">All time bins x all layers</option><option value="focus">One time bin x all layers</option></select></label>
  <label>Time <select id="time"></select></label>
  <button id="show" type="button">Show PNG</button>
</div>
<p id="status" class="status"></p>
<p class="note">Each layer is normalized independently over its eight time bins, so brightness is a spatial distribution within a layer, not a cross-layer importance score. Gray cells are hidden target tokens. Predictor maps show visible-key attention-branch attribution; encoder maps show visible-token attribution from all encoder queries.</p>
<canvas id="canvas"></canvas>
<section id="preview" class="preview" hidden><h2>PNG Preview</h2><img id="previewImage" alt="PNG preview of the current attribution view"></section>
<script>
let A,E;
const refs={},$=id=>refs[id]||(refs[id]=document.getElementById(id)),TARGET='l1';
const MASK_LABELS={temporal_front:'Hide frames 1-2',temporal_middle:'Hide frames 7-8',temporal_back:'Hide frames 15-16',spatial_block:'Continuous spatial mask',spatial_scattered:'Scattered spatial mask'};
let generation=0;
function options(id,values,labels){$(id).replaceChildren(...values.map((value,i)=>{const option=document.createElement('option');option.value=value;option.textContent=(labels||values)[i];return option}))}
function init(){
  const selects=document.querySelectorAll('.controls select');
  refs.scene=selects[0];refs.mask=selects[1];refs.branch=selects[2];refs.view=selects[3];refs.time=selects[4];
  refs.status=document.querySelector('.status');refs.canvas=document.querySelector('canvas');refs.show=document.querySelector('.controls button');refs.preview=document.querySelector('.preview');refs.previewImage=document.querySelector('.preview img');
  const sceneSelect=refs.scene,maskSelect=refs.mask,branchSelect=refs.branch,viewSelect=refs.view,timeSelect=refs.time,statusEl=refs.status,canvasEl=refs.canvas,showButton=refs.show,previewEl=refs.preview,previewImageEl=refs.previewImage;
  options('scene',A.scenes);
  options('mask',A.masks,A.masks.map(mask=>MASK_LABELS[mask]));
  options('time',[0,1,2,3,4,5,6,7],['Frames 1-2','Frames 3-4','Frames 5-6','Frames 7-8','Frames 9-10','Frames 11-12','Frames 13-14','Frames 15-16']);
  maskSelect.value='temporal_back';timeSelect.value='6';
  function draw(){
    document.body.dataset.drawState='entered';
    const gen=++generation,scene=sceneSelect.value,mask=maskSelect.value,branch=branchSelect.value,focus=viewSelect.value==='focus',time=Number(timeSelect.value),stem=scene+'__'+mask;
    const d=A.cases[stem],ed=E.cases[stem],encoder=branch==='encoder',atlas=encoder?ed:d,layerCount=encoder?E.layers:12;
    if(!atlas){statusEl.textContent='Attribution data is unavailable for '+stem+'.';return}
    statusEl.textContent='Loading '+stem+'...';document.body.dataset.drawState='loading';
    const image=new Image();
    image.onerror=()=>{if(gen===generation)statusEl.textContent='Could not load input image: inputs/'+stem+'.jpg';};
    image.onload=()=>{if(gen!==generation)return;try{const ctx=canvasEl.getContext('2d'),size=192,gap=8,left=78,top=104,cols=focus?(encoder?6:4):8;
      if(!ctx)throw new Error('Canvas 2D context is unavailable');
      const rows=focus?Math.ceil(layerCount/cols):layerCount+1;canvasEl.width=left+cols*(size+gap)+18;canvasEl.height=top+rows*(size+(focus?44:gap))+80;
      ctx.fillStyle='#fff';ctx.fillRect(0,0,canvasEl.width,canvasEl.height);ctx.font='15px system-ui';ctx.fillStyle='#1f2933';
      ctx.fillText(scene+' | '+MASK_LABELS[mask]+' | Negative L1 | '+(encoder?'Encoder':'Predictor'),15,25);
      ctx.fillText(focus?timeSelect.selectedOptions[0].text:'All eight time bins',15,50);
      function heatAt(layer,t,index){return encoder?atlas.values[layer][t][index]:atlas.values[layer][0][t][index]}
      function isZero(layer){return encoder?atlas.zero[layer]:atlas.zero[layer][0]}
      function panel(layer,t,x,y){ctx.drawImage(image,t*256,0,256,256,x,y,size,size);for(let i=0;i<256;i++){if(d.hidden[t][i])continue;const q=heatAt(layer,t,i),rgb=A.colors[q];ctx.fillStyle='rgba('+rgb.join(',')+','+(.82*q/255)+')';ctx.fillRect(x+(i%16)*size/16,y+Math.floor(i/16)*size/16,size/16+.2,size/16+.2)}if(isZero(layer)){ctx.fillStyle='#fff';ctx.fillRect(x,y+size/2-16,size,30);ctx.fillStyle='#333';ctx.fillText('Zero positive map',x+36,y+size/2+5)}}
      if(focus){for(let layer=0;layer<layerCount;layer++){const x=left+(layer%cols)*(size+gap),y=top+Math.floor(layer/cols)*(size+44);panel(layer,time,x,y);ctx.fillStyle='#1f2933';ctx.fillText((encoder?'E':'P')+(layer+1),x,y-10)}}
      else{for(let t=0;t<8;t++){ctx.fillStyle='#1f2933';ctx.fillText((2*t+1)+'-'+(2*t+2),left+t*(size+gap),85);ctx.drawImage(image,t*256,0,256,256,left+t*(size+gap),top,size,size)}ctx.fillText('Input',15,top+95);for(let layer=0;layer<layerCount;layer++){const y=top+(layer+1)*(size+gap);ctx.fillText((encoder?'E':'P')+(layer+1),18,y+95);for(let t=0;t<8;t++)panel(layer,t,left+t*(size+gap),y)}}
      ctx.fillStyle='#52606d';ctx.font='13px system-ui';ctx.fillText('Gray: hidden target cells. Heat: positive attribution to visible token slots.',15,canvasEl.height-25);
      const baseline=encoder?ed.baseline:d.baseline[TARGET];statusEl.textContent=scene+' | '+MASK_LABELS[mask]+' | '+(encoder?'Encoder':'Predictor')+' | fixed negative L1 baseline: '+Number(baseline).toFixed(6);
    }catch(error){statusEl.textContent='Render error: '+error.message;console.error(error);}};
    image.src='inputs/'+stem+'.jpg';
  }
  [sceneSelect,maskSelect,branchSelect,viewSelect,timeSelect].forEach(control=>{if(control)control.addEventListener('change',draw)});
  if(showButton)showButton.addEventListener('click',()=>{previewImageEl.src=canvasEl.toDataURL('image/png');previewEl.hidden=false;previewEl.scrollIntoView({behavior:'smooth',block:'start'})});
  draw();
}
window.addEventListener('error',event=>{document.documentElement.dataset.error=(event.message||'script error')+' @'+event.lineno+':'+event.colno;});
function loadScript(src){return new Promise((resolve,reject)=>{const script=document.createElement('script');script.src=src;script.onload=resolve;script.onerror=()=>reject(new Error('Could not load '+src));document.head.appendChild(script)})}
async function boot(){try{await Promise.all([loadScript('data.js'),loadScript('encoder_data.js')]);A=window.ATLAS;E=window.ENCODER_ATLAS;init()}catch(error){$('status').textContent=error.message}}
window.addEventListener('load',boot,{once:true});
</script>
</body>
</html>'''

if __name__=='__main__':main()
