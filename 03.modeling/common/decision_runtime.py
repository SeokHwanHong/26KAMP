"""Dataset-specific decision layer; preserves historical training implementation.

2026-10-03 Claude review revision
- RG3: recheck is driven by 'outside the reference' / information shortage / drift, not by
  every bin that ever saw a risk (the previous rule flagged 100% of records).
  Bin risk history is shown as reference evidence. On development OOF it did not rank
  risk patterns better than chance for RG3, so by default it does not trigger recheck
  (policy rg3_history_triggers_recheck=False; switch on only by explicit agreement).
- Re-sent batches (same content or reused product IDs) are held so one lot cannot count
  as two drift windows. Implemented here so pipeline_runtime.py (LR audit hash) is untouched.
- CN7: an evaluation used to choose among LR/RF/OCSVM cannot also justify promotion.

2026-10-03 second revision (user decision: CN7 = RF + inspection budget, RG3 = drift-based zones)
- CN7: default decision mode 'budget'. Active model (default kind rf) ranks each batch and the top
  inspection_fraction of products is flagged for inspection. CD compares candidates by risk found
  within the same inspection budget instead of fixed-threshold recall/precision.
- RG3: every record gets an uncertainty level (high/medium/low) from drift status + record evidence,
  and the batch gets drift-based uncertainty zones (most-changed variables, out-of-range counts).
  Levels describe how little the model/reference can say, not defect probability.
"""
import math
import hashlib
import numpy as np
import pandas as pd
from pipeline_runtime import (Operations as BaseOperations, range_guidance, write_guidance,
    bins, feature_columns, read, write, digest, clean_id, new_id, now, preprocess, score, topk)

OUTSIDE_TOKENS=('unseen','below_reference_range','above_reference_range')


def usability(metrics, policy):
    reasons=[]
    for key, bound in [('recall','min_recall'),('precision','min_precision')]:
        value=metrics.get(key)
        if value is None or not np.isfinite(value) or value<policy[bound]:
            reasons.append(f'{key} 최소 기준 {policy[bound]} 미달')
    value=metrics.get('FPR')
    if value is None or not np.isfinite(value) or value>policy['max_FPR']:
        reasons.append('오탐률 상한 초과')
    return reasons


def inspection_advice(reference_x, reference_y, products, history_triggers=False):
    """Reference-label evidence only; incoming labels never determine advice.

    evidence_level (priority order)
      out_of_reference         학습 범위 밖 값·미관측 값/조합 → 판단 근거 없음, 추가 검사 권고
      insufficient_information 해당 제품이 걸친 모든 구간이 표본 부족 → 추가 검사 권고
      elevated_risk_history    Wilson 하한이 전체 위험 이력 비율보다 높은 구간에 걸침 → 참고 표시
                               (history_triggers=True일 때만 재검사 권고)
      monitor                  위 해당 없음 → 관찰. 정상 보증 아님
    """
    x=products[feature_columns()]
    unique=x.drop_duplicates().reset_index(drop=True)
    guidance=range_guidance(reference_x,reference_y,unique,None)
    base=float(np.mean(reference_y))
    lookup={(r['variable'],r['interval']):r for r in guidance['ranges']}
    categories={c:bins(x[c],spec) for c,spec in guidance['definitions'].items()}
    for a,b in [('Injection_Time','Filling_Time'),('Max_Injection_Pressure','Mold_Temperature_3')]:
        categories[a+' × '+b]=np.char.add(np.char.add(categories[a],' / '),categories[b])
    rows=[]
    for i in range(len(x)):
        matched=[lookup[(c,str(values[i]))] for c,values in categories.items()]
        outside=[r for r in matched if r['reference_patterns']==0 or any(t in r['interval'] for t in OUTSIDE_TOKENS)]
        risk=[r for r in matched if r['reference_risks']>0]
        elevated=[r for r in risk if r['status']=='descriptive_only'
                  and r['reference_wilson95'][0] is not None and r['reference_wilson95'][0]>base]
        all_sparse=all(r['status']=='insufficient_support' for r in matched)
        if outside:
            level,recheck='out_of_reference',True
            message='학습 자료 범위 밖 조건입니다. 모델·구간 근거가 없으므로 추가 검사로 확인해 주세요.'
        elif all_sparse:
            level,recheck='insufficient_information',True
            message='정보 부족: 안전 여부를 확정할 수 없습니다. 추가 검사로 확인해 주세요.'
        elif elevated:
            level,recheck='elevated_risk_history',bool(history_triggers)
            message=('과거 위험 이력 비율이 평균보다 높게 관측된 구간입니다. '
                     +('직접 재검사를 권고합니다. ' if history_triggers else '참고 정보입니다. ')
                     +'이 구간 정보의 불량 예측력은 검증되지 않았습니다.')
        else:
            level,recheck='monitor',False
            message='분포와 검사 결과를 계속 관찰해 주세요. 정상 보증은 아닙니다.'
        rows.append(dict(record_id=str(products.iloc[i]['record_id']),
            fingerprint=products.iloc[i]['fingerprint'],decision='확정 불가',
            evidence_level=level,reinspection_recommended=recheck,message=message,
            outside_reference=[dict(variable=r['variable'],interval=r['interval']) for r in outside],
            elevated_ranges=[dict(variable=r['variable'],interval=r['interval'],
                patterns=r['reference_patterns'],risks=r['reference_risks'],
                wilson95=r['reference_wilson95']) for r in elevated],
            risk_history_ranges=[dict(variable=r['variable'],interval=r['interval'],
                patterns=r['reference_patterns'],risks=r['reference_risks'],
                observed_rate=r['observed_risk_history_rate'],wilson95=r['reference_wilson95'],
                support=r['status']) for r in risk],
            insufficient_range_count=sum(r['status']=='insufficient_support' for r in matched)))
    return guidance,rows


def batch_drift_advice(drift):
    status=(drift or {}).get('status')
    if status=='review':
        return dict(level='review',message='지속적인 분포 변화: 이 배치 전체의 샘플 검사 비율을 높이고 원인(설비·금형·원료)을 확인해 주세요.')
    if status=='watch':
        return dict(level='watch',message='분포 변화 1회 감지: 다음 배치까지 관찰하고 샘플 검사를 유지해 주세요.')
    if status=='waiting':
        return dict(level='waiting',message='감지 표본 누적 중: 분포 판단을 보류합니다.')
    return dict(level='normal',message='분포 변화 경보 없음. 정상 보증은 아닙니다.')


UNCERTAINTY_TEXT={'high':'높음','medium':'중간','low':'낮음'}


def uncertainty_level(record_level,drift_status):
    """Uncertainty of judgement, NOT defect risk."""
    if drift_status=='review' or record_level=='out_of_reference':return 'high'
    if drift_status=='watch' or record_level=='insufficient_information':return 'medium'
    return 'low'


def drift_zones(drift,top=8):
    """Variables/representations whose distribution moved most in this window (drift output)."""
    changes=(drift or {}).get('changes') or {}
    outside=(drift or {}).get('outside_reference') or {}
    threshold=(drift or {}).get('threshold')
    rows=[dict(representation=k,change_tv=float(v),over_threshold=bool(threshold is not None and v>threshold),
               outside_reference_rows=int(outside.get(k,0)))
          for k,v in sorted(changes.items(),key=lambda kv:-kv[1])[:top]]
    return dict(threshold=threshold,status=(drift or {}).get('status'),zones=rows,
                outside_reference_total={k:int(v) for k,v in outside.items() if v},
                note='분포가 많이 바뀐 변수·조합이다. 불량 원인·불량 범위가 아니다.')


def budget_flags(scores,fraction):
    """Top-k inspection flags for one batch; boundary ties are all flagged (count may exceed k)."""
    s=np.asarray(scores,float);k=max(1,int(math.ceil(len(s)*fraction)))
    cutoff=np.sort(s)[-k];flag=s>=cutoff
    rank=pd.Series(-s).rank(method='min').astype(int).to_numpy()
    return flag,rank,k,float(cutoff)


class Operations(BaseOperations):
    def __init__(self,dataset,state_root=None,policy=None):
        super().__init__(dataset,state_root,policy)
        self.policy.setdefault('min_recall',.5)
        self.policy.setdefault('min_precision',.2)
        self.policy.setdefault('rg3_history_triggers_recheck',False)
        self.policy.setdefault('cn7_decision_mode','budget')      # budget | threshold
        self.policy.setdefault('cn7_model_kind','rf')
        self.policy.setdefault('min_budget_capture',.5)          # share of evaluation risks found within budget
        self.policy.setdefault('acceptance_policy_note','잠정 기준: 현장 검사 비용·누락 허용량 합의 필요')

    # ---- duplicate batch guard (kept outside pipeline_runtime.py) ----
    def _dedup_key(self,frame,batch_id,label_source):
        products,_=preprocess(frame,batch_id,label_source)
        body=products.drop(columns='record_id').sort_values('fingerprint',kind='stable')
        content=hashlib.sha256(pd.util.hash_pandas_object(body,index=False).values.tobytes()).hexdigest()
        explicit=any(c in frame for c in ['record_id','product_id','Unnamed: 0'])
        return content,(set(products.record_id) if explicit else set())

    def _held(self,batch_id,reason):
        folder=self.state/'batches'/batch_id
        with self.lock():folder.mkdir(parents=True,exist_ok=False)
        manifest=dict(id=batch_id,status='held',reason=reason,created_at=now())
        write(folder/'manifest.json',manifest);return manifest

    def ingest(self,frame,batch_id=None,label_source=None,coordinates_confirmed=False):
        batch_id=clean_id(batch_id or new_id('batch'))
        index=read(self.state/'dedup_index.json',{'content':{},'ids':{}})
        key=None
        if coordinates_confirmed:
            try:key=self._dedup_key(frame,batch_id,label_source)
            except (ValueError,TypeError,KeyError):key=None   # base ingest records the precise hold reason
        if key:
            if key[0] in index['content']:
                return self._held(batch_id,f"이미 수집한 배치와 동일한 내용: {index['content'][key[0]]}")
            reused=sorted(key[1]&set(index['ids']))
            if reused:
                return self._held(batch_id,f"이전 배치와 제품 ID 중복 {len(reused)}건: {index['ids'][reused[0]]}")
        result=super().ingest(frame,batch_id,label_source,coordinates_confirmed)
        if result['status']!='accepted':return result
        if key:
            with self.lock():
                index=read(self.state/'dedup_index.json',{'content':{},'ids':{}})
                index['content'][key[0]]=result['id']
                index['ids'].update({i:result['id'] for i in key[1]})
                write(self.state/'dedup_index.json',index)
        if self.dataset=='cn7':return self._cn7_priority(result)
        if self.dataset!='rg3':return result
        folder=self.state/'batches'/result['id']
        baseline,manifest,_=self.baseline()
        rp=baseline/'reference_patterns.csv'
        if digest(rp)!=manifest['patterns_sha256']:raise ValueError('참조 패턴 해시 불일치')
        reference=pd.read_csv(rp,float_precision='round_trip')
        products=pd.read_csv(folder/'products.csv',float_precision='round_trip')
        guidance,rows=inspection_advice(reference[feature_columns()],reference.label,products,
                                        self.policy['rg3_history_triggers_recheck'])
        write_guidance(folder,guidance)
        batch_level=batch_drift_advice(result.get('drift'))
        drift_status=(result.get('drift') or {}).get('status')
        for r in rows:
            u=uncertainty_level(r['evidence_level'],drift_status)
            r['uncertainty_level']=u;r['uncertainty_label']=UNCERTAINTY_TEXT[u]
            r['reference_note']=bool(r['elevated_ranges'])
            if u=='high' and not r['reinspection_recommended']:
                r['reinspection_recommended']=True
                r['message']='지속적인 분포 변화 구간의 제품입니다. 판단 근거가 약하므로 추가 검사를 권고합니다. '+r['message']
        zones=drift_zones(result.get('drift'))
        levels=pd.Series([r['evidence_level'] for r in rows]).value_counts().to_dict()
        uncertainty=pd.Series([r['uncertainty_level'] for r in rows]).value_counts().to_dict()
        advice=dict(dataset='rg3',baseline_id=manifest['id'],created_at=now(),
            drift=result['drift'],batch_advice=batch_level,level_counts=levels,
            uncertainty_counts=uncertainty,uncertainty_zones=zones,
            uncertainty_meaning='높음·중간·낮음은 판단 불확실 정도이며 불량 위험도·불량 확률이 아님',
            history_triggers_recheck=self.policy['rg3_history_triggers_recheck'],
            meaning='검사 권고이며 제품 불량 확률·확정 판정이 아님',
            note=('RG3는 현재 입력으로 위험 패턴을 구분한 근거가 없다(개발 OOF 순위가 무작위와 구분 안 됨). '
                  '재검사 권고는 학습 범위 밖·정보 부족·분포 변화에 근거하며, 구간 위험 이력은 참고 표시다.'),
            records=rows)
        write(folder/'inspection_advice.json',advice)
        flat=pd.DataFrame([{k:v for k,v in r.items() if k not in ('risk_history_ranges','elevated_ranges','outside_reference')}
                           for r in rows])
        flat.to_csv(folder/'inspection_advice.csv',index=False,encoding='utf-8-sig')
        (folder/'inspection_advice.txt').write_text('\n'.join(
            [advice['meaning'],advice['note'],advice['uncertainty_meaning'],'배치 판단: '+batch_level['message'],
             '불확실 단계별 건수: '+str({UNCERTAINTY_TEXT[k]:v for k,v in uncertainty.items()}),
             '근거 단계별 건수: '+str(levels),
             '분포 변화 구간(상위): '+', '.join(f"{z['representation']}({z['change_tv']:.3f})" for z in zones['zones'])]
            +[f"{r['record_id']}: [{r['uncertainty_label']}] {r['message']}" for r in rows]),encoding='utf-8')
        result['inspection_advice']=dict(reinspection_count=sum(r['reinspection_recommended'] for r in rows),
            total=len(rows),batch_advice=batch_level,level_counts=levels,uncertainty_counts=uncertainty,
            message='확정 불가: 학습 범위 밖·정보 부족 항목은 추가 검사, 분포 변화 시 샘플 검사 상향.',
            json=str(folder/'inspection_advice.json'),csv=str(folder/'inspection_advice.csv'),
            text=str(folder/'inspection_advice.txt'),ranges=str(folder/'distribution_guidance.csv'))
        write(folder/'decision_manifest.json',dict(baseline_id=manifest['id'],
            files={p.name:digest(p) for p in folder.glob('*') if p.name.startswith(('inspection_advice.','distribution_guidance.'))}))
        return result

    def _cn7_priority(self,result):
        """CN7 inspection-budget ranking with the active model of policy['cn7_model_kind']."""
        kind=self.policy['cn7_model_kind'];version=self.active().get(kind)
        if self.policy['cn7_decision_mode']!='budget':return result
        if not version:
            result['inspection_priority']=dict(status='no_active_model',
                message=f'CN7 운영 {kind.upper()} 모델이 지정되지 않아 검사 우선순위를 만들지 않았습니다.')
            return result
        folder=self.state/'batches'/result['id']
        products=pd.read_csv(folder/'products.csv',float_precision='round_trip')
        bundle,_=self.load_model(version);scores=score(bundle,products[feature_columns()])
        flag,rank,k,cutoff=budget_flags(scores,self.policy['inspection_fraction'])
        table=pd.DataFrame(dict(record_id=products.record_id.astype(str),fingerprint=products.fingerprint,
            risk_score=scores,rank=rank,inspect=flag.astype(int),model_version=version))
        table=table.sort_values(['rank','record_id'],kind='stable')
        table.to_csv(folder/'inspection_priority.csv',index=False,encoding='utf-8-sig')
        info=dict(status='ranked',model_kind=kind,model_version=version,batch_products=len(products),
            inspection_fraction=self.policy['inspection_fraction'],budget_k=k,flagged=int(flag.sum()),
            score_cutoff=cutoff,
            meaning='배치 안에서 위험 점수 상위 제품을 검사 대상으로 표시. 점수는 위험 이력 패턴 순위이며 불량 확률이 아님',
            note='경계 동점은 모두 포함하므로 표시 수가 예산을 넘을 수 있음. 예산 비율은 현장 검사 여력에 맞춰 사전 확정')
        write(folder/'inspection_priority.json',info)
        (folder/'inspection_priority.txt').write_text('\n'.join([info['meaning'],info['note'],
            f"검사 대상 {info['flagged']}건 / 배치 {len(products)}건 (예산 {k}건, 모델 {version})"]
            +[f"{r.record_id}: 순위 {r.rank}, 점수 {r.risk_score:.4f}" for r in table[table.inspect.eq(1)].itertuples()]),encoding='utf-8')
        write(folder/'decision_manifest.json',dict(files={p.name:digest(p) for p in folder.glob('inspection_priority.*')}))
        result['inspection_priority']=dict(info,csv=str(folder/'inspection_priority.csv'))
        return result

    def _budget_metrics(self,evaluation_id,candidate_bundle,current_bundle):
        p=pd.read_csv(self.state/'evaluations'/clean_id(evaluation_id)/'patterns.csv',float_precision='round_trip')
        y=p.label.to_numpy();x=p[feature_columns()];f=self.policy['inspection_fraction']
        cm=topk(y,score(candidate_bundle,x),f)
        om=topk(y,score(current_bundle,x),f) if current_bundle is not None else None
        return cm,om

    def evaluate(self,candidate,evaluation_id):
        result=super().evaluate(candidate,evaluation_id)
        if self.dataset=='cn7' and result['kind'] in ('lr','rf','ocsvm') and self.policy['cn7_decision_mode']=='budget':
            b,_=self.load_model(candidate);old=self.load_model(result['current'])[0] if result['current'] else None
            cm,om=self._budget_metrics(evaluation_id,b,old)
            result['candidate_budget_metrics']=cm;result['current_budget_metrics']=om
            # Budget mode replaces the fixed-threshold comparison by same-budget risk capture.
            result['reasons']=[r for r in result['reasons'] if r!='F1 개선·재현율 유지·오탐률 상한 조건 미충족']
            enough='평가 정상/위험 고유 패턴 부족' not in result['reasons']
            reachable=min(cm['k'],cm['positives'])   # most risks findable within the budget
            cm['capture_of_reachable']=cm['expected_TP']/reachable if reachable else None
            if enough and cm['expected_TP']<self.policy['min_budget_capture']*reachable:
                result['reasons'].append(f"검사 예산 내 위험 발견 {cm['expected_TP']:.1f}/{reachable}(예산 내 최대) — 최소 비율 {self.policy['min_budget_capture']} 미달")
            if enough and om is not None and cm['expected_TP']<=om['expected_TP']:
                result['reasons'].append('동일 검사량 위험 발견의 개선 없음')
            result['passed']=not result['reasons'];result['decision_mode']='budget'
            result['acceptance_policy_note']=self.policy['acceptance_policy_note']
            folder=self.state/'assessments'/result['id']
            write(folder/'assessment.json',result)
            write(folder/'integrity.json',dict(assessment_sha256=digest(folder/'assessment.json')))
        elif self.dataset=='cn7' and result['kind'] in ('lr','rf','ocsvm'):
            result['reasons'].extend(usability(result['candidate_metrics'],self.policy))
            result['passed']=not result['reasons'];result['decision_mode']='threshold'
            result['acceptance_policy_note']=self.policy['acceptance_policy_note']
            folder=self.state/'assessments'/result['id']
            write(folder/'assessment.json',result)
            write(folder/'integrity.json',dict(assessment_sha256=digest(folder/'assessment.json')))
        return result

    def selection_evaluations(self):
        return {read(p)['evaluation_id'] for p in (self.state/'selections').glob('*.json')}

    def promote(self,assessment_id):
        report=read(self.state/'assessments'/clean_id(assessment_id)/'assessment.json')
        if self.dataset=='cn7' and report and report['kind'] in ('lr','rf','ocsvm'):
            if report.get('decision_mode')=='budget':
                if 'candidate_budget_metrics' not in report:raise ValueError('CN7 검사 예산 평가 기록 없음')
            else:
                reasons=usability(report['candidate_metrics'],self.policy)
                if reasons:raise ValueError('CN7 사용 기준 미충족: '+str(reasons))
        if report and report['evaluation_id'] in self.selection_evaluations():
            raise ValueError('모델 선정에 사용한 평가 자료로는 승격할 수 없음: 별도 평가 자료 등록 필요')
        return super().promote(assessment_id)

    def select_cn7(self,candidates,evaluation_id):
        """Same independent evaluation, qualified candidates only, no activation."""
        if self.dataset!='cn7':raise ValueError('CN7 전용 모델 선정')
        bundles=[self.load_model(v)[0] for v in candidates]
        if len(candidates)!=3 or {b['kind'] for b in bundles}!={'lr','rf','ocsvm'}:
            raise ValueError('LR·RF·OCSVM 후보를 각각 하나씩 제공해야 합니다')
        reports=[dict(self.evaluate(v,evaluation_id)) for v in candidates]
        efolder=self.state/'evaluations'/clean_id(evaluation_id)
        evaluation=None
        if (efolder/'patterns.csv').exists():
            evaluation=pd.read_csv(efolder/'patterns.csv',float_precision='round_trip')
        for report,bundle in zip(reports,bundles):
            # Initial cross-model selection needs no incumbent of every family.
            # Deployment still uses the unchanged assessment and explicit review.
            report['reasons']=[r for r in report['reasons'] if r!='비교할 운영 기준 모델 미지정']
            report['passed']=not report['reasons']
            if evaluation is not None:
                # Inspection-budget view (reference only): CN7 ranks well but fixed thresholds failed.
                report['inspection_budget_reference']=topk(evaluation.label.to_numpy(),
                    score(bundle,evaluation[feature_columns()]),self.policy['inspection_fraction'])
        eligible=[r for r in reports if r['passed']]
        if self.policy['cn7_decision_mode']=='budget':
            # Same budget: more risks found first; ties prefer the simpler/explainable family.
            order={'lr':0,'rf':1,'ocsvm':2}
            eligible.sort(key=lambda r:(-(r.get('candidate_budget_metrics') or r.get('inspection_budget_reference') or {}).get('expected_TP',0),
                                        order.get(r.get('kind'),9),r['candidate']))
        else:
            eligible.sort(key=lambda r:(-r['candidate_metrics']['F1'],-r['candidate_metrics']['recall'],
                                        r['candidate_metrics']['FPR'],r['candidate']))
        result=dict(id=new_id('selection'),dataset='cn7',evaluation_id=evaluation_id,
            status='qualified_candidate' if eligible else 'selection_held',
            selected=eligible[0]['candidate'] if eligible else None,
            reports=reports,policy=self.policy,
            note=('배포하지 않음. 비교 자료는 선택에 사용됐으므로 독립 최종 성능으로 주장하지 않으며, '
                  '같은 평가 자료로는 승격할 수 없음(별도 평가 자료 필요).'))
        write(self.state/'selections'/f"{result['id']}.json",result)
        return result
