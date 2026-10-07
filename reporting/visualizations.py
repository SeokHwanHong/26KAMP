"""Report figures from frozen KAMP results. Never trains, registers or deploys a model.

Run with the operations environment (scikit-learn 1.9.0):
    python reporting/visualizations.py
Notebook report cells call show_notebook(). Historical RF CSVs remain explicitly v1.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, math, sys, os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'tmp/matplotlib_cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, hypergeom
import joblib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/report_visualizations/20261006_v4'
sys.path.insert(0, str(ROOT/'03.modeling/common'))
import pipeline_runtime as rt
from decision_runtime import inspection_advice, budget_flags

plt.rcParams.update({'font.family':'Malgun Gothic', 'axes.unicode_minus':False,
                     'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
BLUE, RED, TEAL = '#38689b', '#bd493f', '#328579'
MANIFEST = []
NOTEBOOKS = {
 '01.EDA/01_eda.ipynb':'eda', '02.preprocessing/02_preprocess.ipynb':'preprocess',
 '02.preprocessing/03_split.ipynb':'split', '03.modeling/04_operations.ipynb':'operations',
 '03.modeling/common/00_pipeline_core.ipynb':'core',
 '03.modeling/models/04.1_isolation_forest.ipynb':'if',
 '03.modeling/models/05_logistic_regression.ipynb':'lr',
 '03.modeling/models/ocsvm.ipynb':'ocsvm', '26kamp.ipynb':'row_order',
 'EDA/cn7_defect_factor_analysis.ipynb':'factors',
 'modeling/random_forest_cn7_integrated.ipynb':'rf_cn7',
 'modeling/random_forest_rg3_integrated.ipynb':'rf_rg3',
 'modeling/relationship_residual_analysis.ipynb':'relationship',
 'modeling/self_training_ocsvm.ipynb':'self_training',
}
VERSIONS = {
 'cn7':{'if':'if-a7a2fcba54e54d58','ocsvm':'ocsvm-6bb288e7f3ba464e',
        'lr':'lr-20261003T040950Z-62637750','rf':'rf-20261005T070541Z-eee06bbc'},
 'rg3':{'if':'if-7e69cab8fdd04a17','ocsvm':'ocsvm-657078ff64fd4278',
        'lr':'lr-20261003T041011Z-d707796f'},
}

def csv(path): return pd.read_csv(ROOT/path)
def save(fig, group, slug, title, context, interpretation, table=None):
    path=OUT/(slug+'.png');fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0,0.025,1,.95))
    fig.savefig(path, dpi=180, facecolor='white');plt.close(fig)
    entry=dict(id=slug,group=group,title=title,context=context,interpretation=interpretation,
               path=path.relative_to(ROOT).as_posix())
    if table is not None:
        data_path=OUT/(slug+'.csv');table.to_csv(data_path,index=False,encoding='utf-8-sig')
        entry['data_path']=data_path.relative_to(ROOT).as_posix()
    MANIFEST.append(entry)

def data():
    result={}
    for d in ('cn7','rg3'):
        x,y,meta,dev,test,_=rt.data(d)
        raw=csv(f'data/origin/moldset_labeled_{d}.csv')
        unl=csv(f'data/origin/moldset_unlabeled_{d}.csv')
        mapping=csv(f'data/processed/{d}/conservative/source_row_mapping.csv')
        assert np.array_equal(mapping.original_label.to_numpy(),raw.PassOrFail.to_numpy())
        audit=csv(f'data/processed/{d}/conservative/labeled_metadata.csv')
        result[d]=dict(x=x,y=y,meta=meta,dev=dev,test=test,raw=raw,unl=unl,mapping=mapping,audit=audit)
    return result

def frozen_predictions(D):
    metrics=[]
    for d,a in D.items():
        test=a['test']; y=a['y'].loc[test].to_numpy(); a['predictions']={}
        for kind,version in VERSIONS[d].items():
            bundle=joblib.load(ROOT/f'runtime/{d}/models/{version}/model.joblib')
            if kind=='if':bundle=dict(bundle,kind='if')
            if kind=='ocsvm': s=rt.library(d)['predict_artifact'](bundle,a['x'].loc[test])[0]
            else:s=rt.score(bundle,a['x'].loc[test])
            threshold=bundle.get('threshold')
            p=None if threshold is None else (np.asarray(s)>threshold).astype(int)
            frame=pd.DataFrame(dict(pattern_row=test,label=y,score=s))
            if p is not None:frame['prediction']=p;frame['threshold']=threshold
            frame.to_csv(OUT/f'{d}_{kind}_frozen_test.csv',index=False,encoding='utf-8-sig')
            a['predictions'][kind]=dict(frame=frame,version=version,threshold=threshold,bundle=bundle)
            k=math.ceil(len(y)*.1); flags=budget_flags(s,.1,ids=[str(v) for v in test],fingerprints=rt.fingerprints(a['x'].loc[test]))
            if isinstance(flags,tuple):flags=flags[0]
            # budget_flags returns its decision table; preserve the runtime tie rule.
            if isinstance(flags,dict):flags=flags['flags']
            row=dict(dataset=d,model=kind,version=version,N=len(y),risk=int(y.sum()),top10_k=k,
                     top10_TP=int(y[np.asarray(flags,dtype=bool)].sum()),threshold=threshold)
            if p is not None:row.update(rt.metrics(y,p))
            metrics.append(row)
    pd.DataFrame(metrics).to_csv(OUT/'frozen_model_metrics.csv',index=False,encoding='utf-8-sig')
    return metrics

def eda_figures(D):
    rows=[]
    for d,a in D.items():
        for unit,y in [('原본 관측치',a['raw'].PassOrFail),('고유 패턴',a['y'])]:
            rows.append(dict(dataset=d.upper(),unit=unit.replace('原본','원본'),normal=int((y==0).sum()),risk=int((y==1).sum())))
    tab=pd.DataFrame(rows);fig,ax=plt.subplots(figsize=(10,5))
    positions=np.arange(4);ax.bar(positions,tab.normal,color=BLUE,label='정상');ax.bar(positions,tab.risk,bottom=tab.normal,color=RED,label='불량 / 위험 이력')
    for i,r in tab.iterrows():ax.text(i,r.normal+r.risk+14,f'N={r.normal+r.risk}\n위험={r.risk}',ha='center')
    ax.set_xticks(positions,[r.dataset+'\n'+r.unit for r in tab.itertuples()]);ax.set_ylim(0,1450);ax.set_ylabel('표본 수');ax.legend()
    save(fig,'eda','eda_counts','원본 관측치와 고유 패턴의 클래스 구성','CN7·RG3 전체 라벨 자료 / 원본 및 패턴 단위 / 모델·임계값 없음',
         '패턴 통합 후에도 위험 클래스가 적다. 정상만 예측하는 정확도는 위험 탐지 성능을 대표하지 않는다.',tab)
    tab=pd.DataFrame([dict(dataset=d.upper(),normal=int((a['audit'].pattern_type=='normal_only').sum()),
            conflicting=int(a['audit'].conflicting.sum()),defect_only=int((a['audit'].pattern_type=='defect_only').sum())) for d,a in D.items()])
    fig,ax=plt.subplots(figsize=(10,4.7));bottom=np.zeros(2)
    for key,label,color in [('normal','정상 전용',BLUE),('conflicting','상충 라벨',RED),('defect_only','불량 전용',TEAL)]:
        ax.barh(tab.dataset,tab[key],left=bottom,color=color,label=label);bottom+=tab[key]
    ax.legend(loc='lower right');ax.set_xlabel('고유 입력 패턴 수')
    for i,r in tab.iterrows():ax.text(10,i,f'정상 {r.normal} / 상충 {r.conflicting} / 불량 전용 {r.defect_only}',va='center',color='white')
    save(fig,'preprocess','preprocess_conflicts','동일 입력에 서로 다른 품질 라벨이 존재한다','전체 라벨 / 패턴 CN7 606·RG3 591 / max(label) 통합 / 모델 없음',
         'CN7 11개, RG3 25개 입력 패턴에서 라벨이 상충하였다. max(label)은 위험 이력을 정의하며 개별 제품의 불량 여부를 확정하지 않는다.',tab)
    fig,axes=plt.subplots(1,2,figsize=(11,5)); ksrows=[]
    for ax,(d,a) in zip(axes,D.items()):
        t=pd.DataFrame([dict(feature=c,KS=ks_2samp(a['raw'].loc[a['raw'].PassOrFail==0,c],a['raw'].loc[a['raw'].PassOrFail==1,c]).statistic) for c in a['x']]).sort_values('KS').tail(8)
        ax.barh(t.feature,t.KS,color=BLUE);ax.set_title(d.upper());ax.set_xlim(0,1);ax.set_xlabel('KS 거리 (기술 통계)');t['dataset']=d;ksrows.append(t)
    save(fig,'eda','eda_feature_separation','품질 라벨별 주요 변수 분포 차이','전체 라벨 원본 CN7 1211·RG3 1182 / 24개 변수 / 탐색 통계, Test 성능 아님',
         'CN7은 위치·배압·금형온도 관련 차이가 관측되었다. 전체 자료의 탐색 결과이므로 독립 검증이나 인과관계로 해석하지 않았다.',pd.concat(ksrows))
    fig,axes=plt.subplots(1,2,figsize=(11,5));corrrows=[]
    for ax,(d,a) in zip(axes,D.items()):
        corr=a['x'].drop(columns=['Clamp_Open_Position']).corr(method='spearman');im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_title(d.upper())
        ax.set_xticks(range(len(corr)),range(1,len(corr)+1));ax.set_yticks(range(len(corr)),range(1,len(corr)+1));ax.tick_params(labelsize=7)
        corr.to_csv(OUT/f'{d}_spearman.csv',encoding='utf-8-sig')
    fig.text(.5,.01,'범례: 파랑 = -1 (음의 상관) / 흰색 = 0 / 빨강 = +1 (양의 상관)',ha='center',fontsize=9)
    save(fig,'eda','eda_correlations','입력 변수의 순위상관 구조 (-1: 음, +1: 양)','전체 고유 패턴 CN7 606·RG3 591 / 상수 열 제외 23개 / 번호는 Clamp_Open_Position을 제외한 schema 열 순서',
         '시간·압력·온도 계열의 중복된 정보가 확인되었다. 높은 상관이 있는 변수를 함께 거리 계산에 사용하면 특정 공정 계열의 영향이 커질 수 있다.')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));rows=[]
    for ax,(d,a) in zip(axes,D.items()):
        for label,col in [(0,BLUE),(1,RED)]:
            q=a['x'].loc[a['y']==label];ax.scatter(q.Max_Back_Pressure,q.Mold_Temperature_3,c=col,s=12 if label==0 else 55,alpha=.5 if label==0 else .9,label=f'{"정상" if label==0 else "위험"} {len(q)}')
        ax.set(xlabel='Max_Back_Pressure (제공 좌표)',ylabel='Mold_Temperature_3 (제공 좌표)',title=d.upper());ax.legend()
    save(fig,'factors','factors_joint','배압과 금형온도의 결합 조건','전체 패턴 CN7 606·RG3 591 / 제공 표준화 좌표 / 탐색 분석, 물리 단위 아님',
         '동일 단변수 구간에서도 다른 공정값에 따라 패턴 구성이 달라진다. 표시된 좌표를 물리적 온도·압력이나 불량 원인으로 확정하지 않았다.')
    a=D['cn7'];cols=['Plasticizing_Position','Max_Injection_Speed','Max_Back_Pressure','Average_Back_Pressure','Mold_Temperature_3','Mold_Temperature_4']
    fig,axes=plt.subplots(3,2,figsize=(11,7.2));q=a['raw'].iloc[:375];bad=q.PassOrFail==1
    for ax,c in zip(axes.ravel(),cols):
        ax.plot(np.arange(len(q)),q[c],lw=.8,color=BLUE);ax.scatter(np.flatnonzero(bad),q.loc[bad,c],color=RED,s=30,zorder=3);ax.set(title=c,xlabel='원본 행 순서 (생산시각 아님)',ylabel='제공 좌표')
    save(fig,'row_order','row_order_cn7','CN7 원본 0–374행의 입력값과 불량 관측 위치','CN7 원본 부분집합 N=375 / 빨강=PassOrFail 1 / 시간축 아님 / 모델 없음',
         '행 순서에 따른 값의 변화를 확인하였다. 생산시각이 없어 시간 추세, 공정 전환 또는 전후 관계로 확정할 수 없다.',q[cols+['PassOrFail']].reset_index(names='source_row'))
    fig,axes=plt.subplots(1,2,figsize=(11,4.7));coverage=[]
    for ax,(d,a) in zip(axes,D.items()):
        t=pd.DataFrame([dict(feature=c,labeled_unique=a['raw'][c].nunique(),unlabeled_unique=a['unl'][c].nunique()) for c in ['Injection_Time','Filling_Time','Clamp_Open_Position','Mold_Temperature_3']]);p=np.arange(len(t));ax.bar(p-.18,t.labeled_unique,.36,label='라벨');ax.bar(p+.18,t.unlabeled_unique,.36,label='비라벨');ax.set_yscale('symlog',linthresh=1);ax.set_xticks(p,t.feature,rotation=30,ha='right');ax.set(title=d.upper(),ylabel='고유 값 수 (로그 축)');ax.legend();t['dataset']=d;coverage.append(t)
    save(fig,'self_training','selftraining_coverage','비라벨 자료의 입력 다양성과 학습 적격성','원본 CN7 라벨1211·비라벨35239 / RG3 라벨1182·비라벨35941 / 비라벨 품질 미확인',
         '비라벨 자료에는 라벨 자료에서 관측되지 않은 값과 더 많은 입력 다양성이 존재한다. 비라벨을 정상으로 간주하거나 자기학습의 성능 개선으로 해석하지 않았다.',pd.concat(coverage))

def split_figures(D):
    fig,axes=plt.subplots(1,2,figsize=(10,4.7));rows=[]
    for ax,(d,a) in zip(axes,D.items()):
        t=pd.DataFrame([dict(partition=s,normal=int((a['y'].loc[idx]==0).sum()),risk=int(a['y'].loc[idx].sum())) for s,idx in [('개발',a['dev']),('Test',a['test'])]])
        ax.bar(t.partition,t.normal,color=BLUE,label='정상');ax.bar(t.partition,t.risk,bottom=t.normal,color=RED,label='위험');ax.set_ylim(0,570);ax.set_title(d.upper())
        for i,r in t.iterrows():ax.text(i,r.normal+r.risk+8,f'{r.normal} / {r.risk}',ha='center')
        ax.legend();t['dataset']=d;rows.append(t)
    save(fig,'split','split_counts','패턴 단위 고정 개발·Test 분할','seed42 / Test20% / CN7 개발484·Test122 / RG3 개발472·Test119',
         '패턴이 두 분할에 중복되지 않도록 고정하였다. Test의 위험 수는 CN7 3개, RG3 5개로 소표본이며 이미 관찰한 후속 평가이다.',pd.concat(rows))
    fig,axes=plt.subplots(1,2,figsize=(10,4.7));rows=[]
    for ax,(d,a) in zip(axes,D.items()):
        assignments=csv(f'data/processed/{d}/conservative/splits/split_assignments.csv');t=assignments.query("partition=='development'").groupby('cv_fold').label.agg(['count','sum']).reset_index()
        ax.bar(t.cv_fold,t['count']-t['sum'],color=BLUE,label='정상');ax.bar(t.cv_fold,t['sum'],bottom=t['count']-t['sum'],color=RED,label='위험');ax.set(title=d.upper(),xlabel='검증 fold',ylabel='고유 패턴 수',ylim=(0,145));ax.set_xticks(range(4));ax.legend(loc='lower left')
        for r in t.itertuples():ax.text(r.cv_fold,r.count+2,f'위험 {r.sum}',ha='center')
        t['dataset']=d;rows.append(t)
    save(fig,'split','split_folds','개발 자료 4-fold의 위험 표본 수','StratifiedKFold 4 / seed42 / 개발 CN7 484·RG3 472 / 모델 없음',
         'CN7의 검증 fold당 위험 수가 2–3개이므로 소수 표본의 탐지 여부가 선택 성능을 크게 바꾼다.',pd.concat(rows))

def score_plot(group,slug,title,panels,context,interpretation):
    fig,axes=plt.subplots(1,len(panels),figsize=(11,4.7),squeeze=False)
    for ax,(name,t,scol,tval) in zip(axes.ravel(),panels):
        y=t['label'].to_numpy();s=t[scol].to_numpy();rng=np.random.default_rng(42)
        for label,color in [(0,BLUE),(1,RED)]:
            mask=y==label;ax.scatter(np.full(mask.sum(),label)+rng.uniform(-.15,.15,mask.sum()),s[mask],s=16 if label==0 else 48,alpha=.45 if label==0 else .95,c=color)
        if tval is not None:ax.axhline(tval,c='black',ls='--',label=f'고정 t={tval:.6g}');ax.legend(fontsize=8)
        ax.set_xticks([0,1],[f'정상 {(y==0).sum()}',f'위험 {(y==1).sum()}']);ax.set(title=name,ylabel='점수 (클수록 위험/이상)');ax.set_xlim(-.5,1.5)
    save(fig,group,slug,title,context,interpretation)

def matrix_plot(group,slug,title,panels,context,interpretation):
    fig,axes=plt.subplots(1,len(panels),figsize=(11,4.2),squeeze=False)
    for ax,(name,m) in zip(axes.ravel(),panels):
        counts=np.array([[m['TN'],m['FP']],[m['FN'],m['TP']]])
        ax.imshow(counts,cmap='Blues');ax.set_xticks([0,1],['정상 예측','위험 예측']);ax.set_yticks([0,1],['실제 정상','실제 위험']);ax.set_title(name)
        for (i,j),v in np.ndenumerate(counts):ax.text(j,i,f'{["TN","FP","FN","TP"][i*2+j]}\n{v}',ha='center',va='center',color='white' if v>counts.max()/2 else 'black',fontsize=12)
    save(fig,group,slug,title,context,interpretation)

def model_figures(D,metrics):
    for kind in ('if','ocsvm','lr'):
        panels=[(d.upper()+' Test',a['predictions'][kind]['frame'],'score',a['predictions'][kind]['threshold']) for d,a in D.items()]
        score_plot(kind,kind+'_scores',kind.upper()+(' 저장 모델의 Test 이상 점수' if kind=='if' else ' 저장 모델의 Test 점수와 고정 임계값'),panels,
          '고유 패턴 CN7 N122·위험3 / RG3 N119·위험5 / '+ ' ; '.join(VERSIONS[d][kind] for d in D),
          'IF는 이상 순위이며 불량 분류 임계값이 없다.' if kind=='if' else '점수 분포와 고정 임계값을 함께 확인하였다. 위험 탐지와 정상 오탐을 별도로 평가하였다.')
        if kind!='if':
            matrix_plot(kind,kind+'_confusion',kind.upper()+' 고정 Test의 FN·FP',[(d.upper(),next(m for m in metrics if m['dataset']==d and m['model']==kind)) for d in D],
             '같은 고정 패턴 Test / 각 저장 모델의 기존 임계값 / Test로 재선택하지 않음',
             'CN7은 위험 3개를 모두 놓쳤다. RG3는 위험 5개를 탐지하였으나 정상 오탐이 많아 임계값 분류의 실용성이 제한되었다.')
        if kind=='lr':
            for d,a in D.items():
                folder=ROOT/f'output/logistic/{d}'/('lr-20261003T040927Z-66e8d685' if d=='cn7' else 'lr-20261003T040950Z-8c8b16ec')
                oof=pd.read_csv(folder/'selected_oof_predictions.csv');test=pd.read_csv(folder/'test_predictions.csv')
                scorecol='probability';score_plot('lr',d+'_lr_oof_test',d.upper()+' LR 개발 OOF와 Test의 구분', [('개발 OOF (선택 자료)',oof,scorecol,float(test.threshold.iloc[0])),('고정 Test (후속 평가)',test,scorecol,float(test.threshold.iloc[0]))],
                    f'{folder.relative_to(ROOT).as_posix()} / 개발{len(oof)}·Test{len(test)} / 기존 선택 임계값',
                    '개발 OOF는 입력 구성·규제 강도·가중치·임계값을 선택한 자료이다. OOF 성능을 독립 일반화 성능으로 표현하지 않았다.')
                coef=pd.read_csv(folder/'coefficients.csv');coef['absolute']=coef.coefficient.abs();t=coef.nlargest(10,'absolute').sort_values('coefficient')
                fig,ax=plt.subplots(figsize=(10,5));ax.barh(t.feature,t.coefficient,color=[RED if z>0 else BLUE for z in t.coefficient]);ax.set_xlabel('LR 계수 (모델 변환 좌표)')
                save(fig,'lr',d+'_lr_coefficients',d.upper()+' 선택 LR의 주요 계수',f'{VERSIONS[d]["lr"]} / 개발 적합 / 상위10 절대계수 / thermal 특징',
                     '계수 부호는 다른 입력을 포함한 모델 좌표에서의 관계이다. 물리적 원인이나 최적 공정값으로 해석하지 않았다.',t)
    for d,a in D.items():
        folder=ROOT/f'output/random_forest_{d}/manual_seven_scenarios_v1';group='rf_'+d
        oof=pd.read_csv(folder/'selected_oof_predictions.csv');test=pd.read_csv(folder/'test_predictions.csv');scol='probability' if 'probability' in test else 'score';tval=float(test.threshold.iloc[0])
        score_plot(group,d+'_rf_v1_scores',d.upper()+' 연구 RF v1의 OOF·Test 점수', [('개발 OOF',oof,scol,tval),('고정 Test',test,scol,tval)],
            f'manual_seven_scenarios_v1 / 개발{len(oof)}·Test{len(test)} / t={tval} / 추가 표준화 제거 이전 결과',
            '기존 v1 결과는 수정 후 RF의 성능이 아니다. 제공 좌표를 직접 사용하는 v2의 전체 후보 탐색 결과는 아직 없다.')
        best=pd.read_csv(folder/'scenario_best.csv');fig,ax=plt.subplots(figsize=(10,4.8));ax.barh(best.scenario[::-1],best.f1[::-1],color=BLUE);ax.set(xlabel='선택 개발 OOF F1',xlim=(0,.65))
        save(fig,group,d+'_rf_v1_scenarios',d.upper()+' 연구 RF v1의 입력 구성별 선택 성능',
            'v1 / 7개 입력 구성 / 개발 OOF F1 / Test 비교·수정 후 성능 아님',
            '각 입력 구성 내에서 설정과 임계값을 선택한 점수이다. 후보 탐색의 선택 편향을 포함하며 Test로 다시 선택하지 않았다.',best)
        importance=pd.read_csv(folder/'impurity_importance.csv').head(10).iloc[::-1];fig,ax=plt.subplots(figsize=(10,5));ax.barh(importance.feature,importance.impurity_importance,color=TEAL);ax.set_xlabel('불순도 감소 중요도')
        save(fig,group,d+'_rf_v1_importance',d.upper()+' 연구 RF v1의 변수 중요도',
             'v1 최종 개발 적합 / 상위10 / 수정 후 RF 성능과 구분',
             '중요도는 해당 모델의 분기 사용 정도를 나타낸다. 상관된 입력 사이에 중요도가 나뉠 수 있으며 인과적 불량 요인이 아니다.',importance)
    # Same-budget model comparison uses exactly the same frozen Test patterns.
    fig,axes=plt.subplots(1,2,figsize=(11,4.8));curve=[]
    for ax,(d,a) in zip(axes,D.items()):
        y=a['y'].loc[a['test']].to_numpy();fps=rt.fingerprints(a['x'].loc[a['test']]);fracs=np.arange(.05,.51,.05)
        for kind,obj in a['predictions'].items():
            found=[]
            for f in fracs:
                flags=budget_flags(obj['frame'].score,f,ids=[str(v) for v in a['test']],fingerprints=fps)[0]
                tp=int(y[flags].sum());found.append(tp);curve.append(dict(dataset=d,model=kind,fraction=f,k=math.ceil(len(y)*f),TP=tp,recall=tp/y.sum()))
            ax.plot(fracs*100,found,'o-',label=kind.upper())
        ax.plot(fracs*100,[math.ceil(len(y)*f)*y.mean() for f in fracs],'--',color='black',label='무작위 기대값');ax.set(title=d.upper(),xlabel='검사 비율 (%)',ylabel='발견 위험 패턴 수',ylim=(-.2,int(y.sum())+.4));ax.legend()
    save(fig,'if','model_budget_curves','같은 검사 예산에서 저장 모델의 위험 발견 수','고정 패턴 Test CN7 N122·위험3 / RG3 N119·위험5 / exact_k / 5–50% / 임계값 분류와 별도',
         'CN7의 10% 검사량에서는 고정 RF가 3개, LR이 2개를 포함하였다. RG3에서는 IF가 2개를 포함하였다. 소표본의 후속 결과이며 현장 효과로 확정하지 않았다.',pd.DataFrame(curve))

def error_figures(D):
    rows=[]
    for d,a in D.items():
        kind='rf' if d=='cn7' else 'ocsvm';obj=a['predictions'][kind];t=obj['frame'].set_index('pattern_row');q=a['x'].loc[a['test']].copy();q['label']=t.label;q['prediction']=t.prediction;q['score']=t.score
        q['category']=np.select([(q.label==1)&(q.prediction==1),(q.label==0)&(q.prediction==1),(q.label==1)&(q.prediction==0)],['TP','FP','FN'],default='TN');q['pattern_row']=q.index
        q.to_csv(OUT/f'{d}_actual_error_cases.csv',index=False,encoding='utf-8-sig')
        cols=['Max_Back_Pressure','Max_Injection_Pressure','Cushion_Position','Max_Injection_Speed','Mold_Temperature_3','Mold_Temperature_4']
        normal=a['x'].loc[a['dev']].loc[a['y'].loc[a['dev']]==0];center=normal[cols].median();scale=(normal[cols].quantile(.75)-normal[cols].quantile(.25)).replace(0,1)
        selected=q.loc[(q.label==1)|(q.category=='FP')].sort_values(['label','score'],ascending=[False,False]).head(15)
        z=(selected[cols]-center)/scale;fig,ax=plt.subplots(figsize=(11,5.8));im=ax.imshow(z,cmap='RdBu_r',vmin=-3,vmax=3,aspect='auto');ax.set_xticks(range(len(cols)),cols,rotation=25,ha='right');ax.set_yticks(range(len(z)),[f'{r.category} #{r.pattern_row} s={r.score:.3g}' for r in selected.itertuples()]);fig.colorbar(im,ax=ax,label='(값 - 개발 정상 중앙값) / IQR')
        save(fig,'factors' if d=='cn7' else 'ocsvm',d+'_error_conditions',d.upper()+' 실제 위험·오탐 패턴의 공정값',
             f'{obj["version"]} / 고정 Test / t={obj["threshold"]:.6g} / CN7 전체 위험3+FP8; RG3 위험5+상위점수 FP10 / 표시만 IQR 변환',
             '오류 사례의 실제 입력과 점수를 대조하였다. 색상은 시각화용 상대 위치이며 모델 입력을 다시 표준화한 것이 아니다. RG3의 오탐 전수는 별도 CSV에 보존하였다.',selected)
        fig,axes=plt.subplots(1,2,figsize=(11,4.7));counts=[]
        for ax,(c1,c2) in zip(axes,[('Max_Back_Pressure','Mold_Temperature_3'),('Max_Injection_Pressure','Cushion_Position')]):
            for category,color,marker in [('TN',BLUE,'.'),('FP','#de9636','x'),('FN',RED,'x'),('TP',TEAL,'o')]:
                s=q[q.category==category];ax.scatter(s[c1],s[c2],c=color,marker=marker,s=30 if category!='TN' else 16,alpha=.8,label=f'{category} {len(s)}')
            ax.set(xlabel=c1,ylabel=c2);ax.legend(fontsize=8)
        save(fig,'factors' if d=='cn7' else 'ocsvm',d+'_error_joint',d.upper()+' TP·TN·FN·FP의 두 변수 결합 분포',
             f'{obj["version"]} / 고정 Test N={len(q)} / 제공 좌표 / 동일 고정 임계값',
             '성공·실패 사례가 함께 분포하는 구간을 확인하였다. Test에서 관측한 구간을 새 의사결정 경계로 사용하지 않았다.',q[['pattern_row',*cols,'label','prediction','score','category']])
        # Fixed development-normal quartile boundaries; report zero-support bins too.
        c='Max_Back_Pressure';bounds=np.unique(normal[c].quantile([.25,.5,.75]));bins=np.r_[-np.inf,bounds,np.inf];q['interval']=pd.cut(q[c],bins=bins).astype(str)
        tab=q.groupby('interval').agg(N=('label','size'),risk=('label','sum'),FP=('category',lambda v:int((v=='FP').sum())),FN=('category',lambda v:int((v=='FN').sum()))).reset_index();tab['dataset']=d;rows.append(tab)
    pd.concat(rows).to_csv(OUT/'error_interval_counts.csv',index=False,encoding='utf-8-sig')
    joint_rows=[]
    for d,a in D.items():
        q=pd.read_csv(OUT/f'{d}_actual_error_cases.csv')
        normal=a['x'].loc[a['dev']].loc[a['y'].loc[a['dev']]==0]
        for c in ['Max_Back_Pressure','Mold_Temperature_3']:
            cutoff=float(normal[c].median());q[c+'_region']=np.where(q[c]<=cutoff,'하위','상위')
        for first in ['하위','상위']:
            for second in ['하위','상위']:
                part=q[(q.Max_Back_Pressure_region==first)&(q.Mold_Temperature_3_region==second)]
                joint_rows.append(dict(dataset=d,back_pressure=first,mold_temperature=second,N=len(part),risk=int(part.label.sum()),
                    TP=int((part.category=='TP').sum()),FP=int((part.category=='FP').sum()),FN=int((part.category=='FN').sum()),TN=int((part.category=='TN').sum()),
                    pressure_cutoff=float(normal.Max_Back_Pressure.median()),temperature_cutoff=float(normal.Mold_Temperature_3.median())))
    pd.DataFrame(joint_rows).to_csv(OUT/'error_joint_counts.csv',index=False,encoding='utf-8-sig')

def relationship_figures():
    for d in ('cn7','rg3'):
        folder=ROOT/f'output/relationship_residual/20261001_relations_v1/{d}_fixed_pattern'
        tab=pd.read_csv(folder/'relationship_oof_quality.csv');t=tab.groupby(['backend','target']).oof_r2.mean().unstack(0).dropna(how='all').sort_index()
        fig,ax=plt.subplots(figsize=(11,7));t.plot.barh(ax=ax);ax.axvline(.2,color='black',ls='--');ax.set_xlabel('정상 관계식 OOF R² 평균');ax.legend(title='예측기')
        save(fig,'relationship',d+'_relationship_quality',d.upper()+' 정상 변수 관계식의 예측 가능성',
            '20261001_relations_v1 / fixed_pattern / 정상 reference 내부 GroupKFold / 유지 기준 R²≥0.2 / 품질 판별 성능 아님',
            '다른 공정값으로 각 입력을 예측하는 관계식의 정확성을 확인하였다. 정상 관계의 예측력이 높아도 위험 판별 성능이 높다는 뜻은 아니다.',t.reset_index())
        tab=pd.read_csv(folder/'test_comparison.csv');fig,ax=plt.subplots(figsize=(11,5.3));colors=[TEAL if v else BLUE for v in tab.selected_on_validation];ax.barh(tab.model,tab.F1,color=colors);ax.set(xlabel='고정 Test F1',xlim=(0,.6))
        save(fig,'relationship',d+'_relationship_comparison',d.upper()+' 관계 잔차 실험의 모든 모델 Test 결과',
             f'fixed_pattern / Test CN7 122·위험3, RG3 119·위험5 / 각 검증 고정 임계값 / 초록=검증 선택 모델',
             '원본 입력, 잔차 입력, 결합 입력과 비지도 점수를 동일 실험 안에서 비교하였다. 유리한 후보만 발췌하지 않았으며 기존 Test의 후속 평가이다.',tab)
    fig,axes=plt.subplots(1,2,figsize=(11,5.4));tabs=[]
    for ax,d in zip(axes,('cn7','rg3')):
        folder=ROOT/f'output/stacking_complementarity/20261001_nested_v1/{d}';tab=pd.read_csv(folder/'test_metrics.csv');p=np.arange(len(tab));ax.bar(p-.18,tab.top10_expected_TP,.36,label='Test 10% 발견수');dev=pd.read_csv(folder/'validation_metrics.csv').set_index('model');ax.bar(p+.18,[dev.loc[m,'top10_expected_TP'] for m in tab.model],.36,label='개발 OOF 발견수');ax.set_xticks(p,tab.model,rotation=35,ha='right');ax.set(title=d.upper(),ylabel='발견 위험 패턴 수');ax.legend(fontsize=8);tab['dataset']=d;tabs.append(tab)
    save(fig,'relationship','stacking_same_experiment','동일 nested 실험 안에서 단일 모델과 결합 모델 비교',
         '20261001_nested_v1 / 외부4·내부3 fold / 개발 CN7 484·RG3 472 / Test 122·119 / 10%·동점 기대값',
         'CN7 스태킹은 개발 11/11, Test 3/3을 포함하였다. RG3는 개발 4/20, Test 0/5로 이어졌다. 고정 운영 RF·IF와 다른 학습 설정이므로 성능을 합치지 않았다.',pd.concat(tabs))

def operations_figures(D):
    # Apply the actual governance method to isolated files, bypassing only the
    # parent's advice-writing step because that evidence was computed above.
    from governance_runtime import Operations
    from workflow_runtime import Operations as Parent
    from unittest.mock import patch
    a=D['rg3'];mapping=a['mapping'];rawtest=mapping[mapping.pattern_row.isin(a['test'])].copy()
    products=a['raw'].iloc[rawtest.source_row][rt.feature_columns()].reset_index(drop=True)
    products['record_id']=['source:'+str(i) for i in rawtest.source_row];products['fingerprint']=rt.fingerprints(products[rt.feature_columns()])
    guidance,records=inspection_advice(a['x'].loc[a['dev']],a['y'].loc[a['dev']],products)
    labels=rawtest.original_label.to_numpy();assert len(labels)==238 and labels.sum()==5
    folder=OUT/'isolated_rg3_policy/batches/report-rg3-test';folder.mkdir(parents=True,exist_ok=True)
    results=[];plans=[]
    for fraction in (.05,.1,.2):
        rt.write(folder/'inspection_advice.json',dict(records=records));pd.DataFrame(records).to_csv(folder/'inspection_advice.csv',index=False);(folder/'inspection_advice.txt').write_text('後향 모의평가\n'.replace('後','후'),encoding='utf-8');rt.write(folder/'decision_manifest.json',dict(files={}))
        obj=Operations.__new__(Operations);obj.state=OUT/'isolated_rg3_policy';obj.policy=dict(governance_version='report-fixed-policy',rg3_sampling={s:dict(fraction=fraction,random_share=.5) for s in ['waiting','normal','watch','review']})
        with patch.object(Parent,'_rg3_advice',return_value={'inspection_advice':{}}):
            answer=obj._rg3_advice(dict(id='report-rg3-test',drift={'status':'waiting'}))
        plan=answer['inspection_plan'];plans.append(plan);ids=products.record_id.tolist();pos={v:i for i,v in enumerate(ids)}
        selected=[pos[v] for v in plan['selected_ids']];priority=[pos[v] for v in plan['priority_ids']];remain=np.array([i for i in range(len(ids)) if i not in priority]);k=plan['budget_k'];random_k=len(plan['random_ids']);rng=np.random.default_rng(42)
        repeated=np.array([labels[priority].sum()+labels[rng.choice(remain,size=random_k,replace=False)].sum() for _ in range(1000)])
        policy_expected=float(labels[priority].sum()+random_k*labels[remain].mean())
        pure=hypergeom(len(labels),int(labels.sum()),k);if_scores=a['predictions']['if']['frame'].set_index('pattern_row').score;rawscores=rawtest.pattern_row.map(if_scores).to_numpy();flags=budget_flags(rawscores,fraction,ids=ids,fingerprints=products.fingerprint.tolist())[0]
        results.append(dict(fraction=fraction,k=k,policy_fixed_TP=int(labels[selected].sum()),policy_mean_TP=policy_expected,policy_sim_mean=float(repeated.mean()),policy_p025=float(np.quantile(repeated,.025)),policy_p975=float(np.quantile(repeated,.975)),IF_TP=int(labels[flags].sum()),random_mean_TP=float(pure.mean()),random_p025=float(pure.ppf(.025)),random_p975=float(pure.ppf(.975)),priority_k=len(priority),random_k=random_k))
    rt.write(OUT/'rg3_actual_policy_plans.json',plans);pd.DataFrame(records).to_csv(OUT/'rg3_policy_evidence.csv',index=False,encoding='utf-8-sig')
    tab=pd.DataFrame(results);fig,ax=plt.subplots(figsize=(11,5));p=np.arange(len(tab));w=.24
    ax.bar(p-w,tab.IF_TP,w,label='IF 单독 순위'.replace('单','단'),color=BLUE);ax.bar(p,tab.policy_mean_TP,w,label='실제 우선+무작위 규칙 기대값',color=TEAL);ax.errorbar(p,tab.policy_mean_TP,yerr=[tab.policy_mean_TP-tab.policy_p025,tab.policy_p975-tab.policy_mean_TP],fmt='none',color='black',capsize=4)
    ax.bar(p+w,tab.random_mean_TP,w,label='단순 무작위 기대값',color='#929aa5');ax.errorbar(p+w,tab.random_mean_TP,yerr=[tab.random_mean_TP-tab.random_p025,tab.random_p975-tab.random_mean_TP],fmt='none',color='black',capsize=4)
    ax.set_xticks(p,[f'{f:.0%}\n{k}행 검사' for f,k in zip(tab.fraction,tab.k)]);ax.set(ylabel='발견 불량 원본 관측치 수',ylim=(0,5.5));ax.legend(fontsize=9)
    save(fig,'operations','rg3_actual_policy_comparison','RG3 실제 정책 규칙과 IF·무작위의 동일 예산 비교',
         '고정 Test 대응 원본 N238·불량5 / 정책 함수 실제 호출·격리 폴더 / random_share0.5 / 1000회·seed42 / 수직선95% 변동 범위',
         '정책 우선순위는 참조 범위 이탈·정보 부족과 ID 순서로 결정하였다. 평가 라벨은 대상 선정에 사용하지 않았다. 비율5·10·20%는 사전 예시이며 승인된 현장 정책이나 최적 예산이 아니다.',tab)
    # Real distribution comparison, without calling row blocks production batches.
    fig,axes=plt.subplots(2,2,figsize=(11,6.8));rows=[]
    for row,(d,a) in enumerate(D.items()):
        for col,c in enumerate(['Filling_Time','Mold_Temperature_3']):
            ax=axes[row,col];ref=a['x'].loc[a['dev']].loc[a['y'].loc[a['dev']]==0,c];new=a['unl'][c];edges=np.unique(np.quantile(np.r_[ref,new],np.linspace(0,1,35)));ax.hist(ref,bins=edges,weights=np.ones(len(ref))/len(ref),alpha=.6,label=f'개발 정상 {len(ref)}',color=BLUE);ax.hist(new,bins=edges,weights=np.ones(len(new))/len(new),alpha=.45,label=f'비라벨 {len(new)}',color=RED);ax.set(title=d.upper()+' '+c,ylabel='집단 내 비율',xlabel='제공 좌표');ax.legend(fontsize=8);rows.append(dict(dataset=d,variable=c,reference_n=len(ref),unlabeled_n=len(new),KS=ks_2samp(ref,new).statistic))
    save(fig,'operations','unlabeled_distribution_change','개발 정상 참조와 실제 비라벨 자료의 입력 분포',
         'CN7 참조473·비라벨35239 / RG3 참조452·비라벨35941 / 제공 좌표 / 품질 라벨·생산시각 없음',
         '실제 비라벨 분포 차이를 확인하였다. 두 자료의 최초 표준화 기준 정합이 확인되지 않아 현장 드리프트 또는 품질 악화로 확정하지 않았다.',pd.DataFrame(rows))
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,(d,a) in zip(axes,D.items()):
        obj=a['predictions']['if'];bundle=obj['bundle'];ref=rt.score(bundle,a['x'].loc[a['dev']].loc[a['y'].loc[a['dev']]==0]);new=rt.score(bundle,a['unl'][rt.feature_columns()]);bins=np.linspace(min(ref.min(),new.min()),max(ref.max(),new.max()),40)
        ax.hist(ref,bins=bins,weights=np.ones(len(ref))/len(ref),alpha=.65,color=BLUE,label='개발 정상');ax.hist(new,bins=bins,weights=np.ones(len(new))/len(new),alpha=.45,color=RED,label='비라벨');ax.set(title=d.upper(),xlabel='고정 IF 이상 점수',ylabel='집단 내 비율');ax.legend()
        pd.DataFrame({'score':new}).to_csv(OUT/f'{d}_unlabeled_if_scores.csv',index=False,encoding='utf-8-sig')
    save(fig,'operations','unlabeled_score_change','같은 IF 모델에서 참조·비라벨 점수 분포 비교',
         '고정 IF CN7 if-a7a2fcba54e54d58·RG3 if-7e69cab8fdd04a17 / 비라벨 전수 / 모델·좌표 기준 고정',
         '비라벨의 점수 이동은 같은 모델에서 비교하였다. 불량 라벨이 없어 점수 변화만으로 품질 악화나 재학습 효과를 판정할 수 없다.')
    return tab

def flow(group,slug,title,boxes,context,interpretation):
    fig,ax=plt.subplots(figsize=(11,5.3));ax.axis('off')
    n=len(boxes)
    for i,(head,body) in enumerate(boxes):
        col=i%3;row=i//3;x=.025+col*.333;y=.73-row*.4
        ax.text(x,y,head+'\n\n'+body,transform=ax.transAxes,va='top',fontsize=10,bbox=dict(boxstyle='round,pad=.65',facecolor='#edf3f7',edgecolor=BLUE),linespacing=1.45)
        if col<2 and i<n-1:ax.annotate('',xy=(x+.315,y-.1),xytext=(x+.275,y-.1),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.6))
    save(fig,group,slug,title,context,interpretation)

def flow_figures():
    flow('core','core_data_contract','입력 계약과 패턴·원본 연결',[
       ('1. 입력 검증','24개 수치형 열·순서\n유한값 / 품질 라벨 분리'),('2. 정확한 패턴 통합','동일 입력 / max(label)\n원본 빈도와 상충 이력 유지'),('3. 분할 고정','Test20% / 개발4-fold\n동일 패턴 교차 포함 차단'),('4. 모델 저장','특징 변환·모델·임계값\n버전 / 입력 계약 / 해시'),('5. 추론 결과','패턴 점수 → 원본 연결\n검사 예산 exact_k'),('6. 근거 보존','결정·정책·라벨 출처\n변경 승인과 감사 기록')],
       '현행 공통 코드와 저장 계약 / 절차도 / 성능 결과 아님',
       '패턴 학습과 원본 관측치 검사 평가를 연결하였다. 품질 라벨, 원본 ID와 빈도는 예측 입력에 포함하지 않았다.')
    flow('operations','operations_ct_decision','변화 감지부터 재학습·승격까지의 판단 절차',[
       ('1. 분포 감지','입력·점수 TV / bootstrap\n표본 충분성 / 연속 변화'),('2. 원인 확인','입력 오류·좌표 정합\n공정 변경·정상 공정 확인'),('3. 실제 라벨 확보','생산시각·공정 버전\n검사 출처 / 중복 차단'),('4. 승인 후보 적합','기준 설정·임계값 유지\n학습과 평가 패턴 분리'),('5. 같은 평가 자료 비교','기존·후보 FN/FP\n같은 검사 예산 / 불량수'),('6. 승인 또는 유지','기준 충족 시 승격\n미충족 유지 / 롤백')],
       'governance_runtime·workflow_runtime / 구현 절차도 / 독립 현장 재학습 전후 결과 없음',
       '변화 감지는 재학습 승인 사유의 일부이다. 실제 독립 holdout의 기존·후보 성능 자료가 없으므로 개선·유지 사례를 실증 성과로 만들지 않았다.')
    flow('self_training','selftraining_status','자기학습 노트북과 현행 승인 학습의 적용 상태',[
       ('비라벨 입력','CN7 35239 / RG3 35941\n정상 여부 미확인'),('기존 자기학습 초안','이전 selection_manifest 필요\n참조 모델 경로 미존재'),('실행 결과','pseudo-label 반복 학습\n독립 개선 결과 없음'),('현행 운영 원칙','실제 검사 라벨·출처 확보\n좌표 정합과 원인 확인'),('CT 승인','독립 평가 계획·승인\n기존 모델 자동 변경 없음'),('품질 판정','같은 holdout·검사량 비교\n검증 전 효과 확정 불가')],
       'self_training_ocsvm.ipynb의 현행 의존 경로 조사 / 상태·절차도 / 성능 그래프 아님',
       '자기학습 초안의 산출물이 없어 성능 그래프를 생성하지 않았다. 비라벨 자료를 자동 정상 라벨로 바꾸는 방식은 현행 승인 운영과 구분하였다.')

def generate_all():
    OUT.mkdir(parents=True,exist_ok=True);MANIFEST.clear();D=data();metrics=frozen_predictions(D)
    eda_figures(D);split_figures(D);model_figures(D,metrics);error_figures(D);relationship_figures();policy=operations_figures(D);flow_figures()
    rt.write(OUT/'manifest.json',dict(figures=MANIFEST,notebooks=NOTEBOOKS,model_versions=VERSIONS,
             helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             evaluation='historical fixed Test follow-up; not new independent field validation',
             model_training_performed=False,runtime_modified=False))
    print(f'Generated {len(MANIFEST)} figures for {len(NOTEBOOKS)} notebooks')
    return MANIFEST

def show_notebook(notebook_path):
    """Display this notebook's complete report figures; no old training cells run."""
    from IPython.display import Image, Markdown, display
    key=str(notebook_path).replace('\\','/');group=NOTEBOOKS[key]
    manifest=OUT/'manifest.json'
    if not manifest.exists():generate_all()
    obj=json.loads(manifest.read_text(encoding='utf-8'))
    if obj['helper_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError('시각화 코드가 변경되었습니다. python reporting/visualizations.py를 먼저 실행하세요.')
    figures=[f for f in obj['figures'] if f['group']==group]
    if not figures:raise RuntimeError(f'No report figures for {key}')
    for f in figures:
        display(Markdown(f'### {f["title"]}\n\n{f["context"]}\n\n{f["interpretation"]}'))
        display(Image(filename=str(ROOT/f['path'])))
    return [f['id'] for f in figures]

if __name__=='__main__':generate_all()
