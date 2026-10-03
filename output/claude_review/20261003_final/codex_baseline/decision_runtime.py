"""Dataset-specific decision layer; preserves historical training implementation."""
import numpy as np
import pandas as pd
from pipeline_runtime import (Operations as BaseOperations, range_guidance, write_guidance,
    bins, feature_columns, read, write, digest, clean_id, new_id, now)


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


def inspection_advice(reference_x, reference_y, products):
    """Reference-label evidence only; incoming labels never determine advice."""
    x=products[feature_columns()]
    unique=x.drop_duplicates().reset_index(drop=True)
    guidance=range_guidance(reference_x,reference_y,unique,None)
    lookup={(r['variable'],r['interval']):r for r in guidance['ranges']}
    categories={c:bins(x[c],spec) for c,spec in guidance['definitions'].items()}
    for a,b in [('Injection_Time','Filling_Time'),('Max_Injection_Pressure','Mold_Temperature_3')]:
        categories[a+' × '+b]=np.char.add(np.char.add(categories[a],' / '),categories[b])
    rows=[]
    for i in range(len(x)):
        matched=[lookup[(c,str(values[i]))] for c,values in categories.items()]
        risk=[r for r in matched if r['reference_risks']>0]
        sparse=[r for r in matched if r['status']=='insufficient_support']
        # A historical risk is evidence for checking, never a defect verdict.
        level='risk_history_recheck' if risk else 'insufficient_information' if sparse else 'monitor'
        rows.append(dict(record_id=str(products.iloc[i]['record_id']),
            fingerprint=products.iloc[i]['fingerprint'],decision='확정 불가',
            evidence_level=level,reinspection_recommended=bool(risk or sparse),
            message=('과거 위험 이력이 있는 구간입니다. 정상·불량 확정이 어려우므로 직접 재검사를 권고합니다.'
                     if risk else '정보 부족: 안전 여부를 확정할 수 없습니다. 추가 검사로 확인해 주세요.'
                     if sparse else '분포와 검사 결과를 계속 관찰해 주세요. 정상 보증은 아닙니다.'),
            risk_history_ranges=[dict(variable=r['variable'],interval=r['interval'],
                patterns=r['reference_patterns'],risks=r['reference_risks'],
                observed_rate=r['observed_risk_history_rate'],wilson95=r['reference_wilson95'],
                support=r['status']) for r in risk],
            insufficient_range_count=len(sparse)))
    return guidance,rows


class Operations(BaseOperations):
    def __init__(self,dataset,state_root=None,policy=None):
        super().__init__(dataset,state_root,policy)
        self.policy.setdefault('min_recall',.5)
        self.policy.setdefault('min_precision',.2)
        self.policy.setdefault('acceptance_policy_note','잠정 기준: 현장 검사 비용·누락 허용량 합의 필요')

    def ingest(self,*args,**kwargs):
        result=super().ingest(*args,**kwargs)
        if self.dataset!='rg3' or result['status']!='accepted':return result
        folder=self.state/'batches'/result['id']
        baseline,manifest,_=self.baseline()
        rp=baseline/'reference_patterns.csv'
        if digest(rp)!=manifest['patterns_sha256']:raise ValueError('참조 패턴 해시 불일치')
        reference=pd.read_csv(rp,float_precision='round_trip')
        products=pd.read_csv(folder/'products.csv',float_precision='round_trip')
        guidance,rows=inspection_advice(reference[feature_columns()],reference.label,products)
        write_guidance(folder,guidance)
        advice=dict(dataset='rg3',baseline_id=manifest['id'],created_at=now(),
            drift=result['drift'],meaning='검사 권고이며 제품 불량 확률·확정 판정이 아님',
            note='드리프트가 없어도 위험 이력·정보 부족에 따른 검사 권고는 유지됩니다.',
            records=rows)
        write(folder/'inspection_advice.json',advice)
        flat=pd.DataFrame([{k:v for k,v in r.items() if k!='risk_history_ranges'} for r in rows])
        flat.to_csv(folder/'inspection_advice.csv',index=False,encoding='utf-8-sig')
        (folder/'inspection_advice.txt').write_text('\n'.join(
            [advice['meaning'],advice['note']]+[f"{r['record_id']}: {r['message']}" for r in rows]),encoding='utf-8')
        result['inspection_advice']=dict(reinspection_count=sum(r['reinspection_recommended'] for r in rows),
            total=len(rows),message='확정 불가: 위험 이력 또는 정보 부족 항목은 직접 재검사해 주세요.',
            json=str(folder/'inspection_advice.json'),csv=str(folder/'inspection_advice.csv'),
            text=str(folder/'inspection_advice.txt'),ranges=str(folder/'distribution_guidance.csv'))
        write(folder/'decision_manifest.json',dict(baseline_id=manifest['id'],
            files={p.name:digest(p) for p in folder.glob('*') if p.name.startswith(('inspection_advice.','distribution_guidance.'))}))
        return result

    def evaluate(self,candidate,evaluation_id):
        result=super().evaluate(candidate,evaluation_id)
        if self.dataset=='cn7' and result['kind'] in ('lr','rf','ocsvm'):
            result['reasons'].extend(usability(result['candidate_metrics'],self.policy))
            result['passed']=not result['reasons']
            result['acceptance_policy_note']=self.policy['acceptance_policy_note']
            folder=self.state/'assessments'/result['id']
            write(folder/'assessment.json',result)
            write(folder/'integrity.json',dict(assessment_sha256=digest(folder/'assessment.json')))
        return result

    def promote(self,assessment_id):
        report=read(self.state/'assessments'/clean_id(assessment_id)/'assessment.json')
        if self.dataset=='cn7' and report and report['kind'] in ('lr','rf','ocsvm'):
            reasons=usability(report['candidate_metrics'],self.policy)
            if reasons:raise ValueError('CN7 사용 기준 미충족: '+str(reasons))
        return super().promote(assessment_id)

    def select_cn7(self,candidates,evaluation_id):
        """Same independent evaluation, qualified candidates only, no activation."""
        if self.dataset!='cn7':raise ValueError('CN7 전용 모델 선정')
        bundles=[self.load_model(v)[0] for v in candidates]
        if len(candidates)!=3 or {b['kind'] for b in bundles}!={'lr','rf','ocsvm'}:
            raise ValueError('LR·RF·OCSVM 후보를 각각 하나씩 제공해야 합니다')
        reports=[dict(self.evaluate(v,evaluation_id)) for v in candidates]
        for report in reports:
            # Initial cross-model selection needs no incumbent of every family.
            # Deployment still uses the unchanged assessment and explicit review.
            report['reasons']=[r for r in report['reasons'] if r!='비교할 운영 기준 모델 미지정']
            report['passed']=not report['reasons']
        eligible=[r for r in reports if r['passed']]
        eligible.sort(key=lambda r:(-r['candidate_metrics']['F1'],-r['candidate_metrics']['recall'],
                                    r['candidate_metrics']['FPR'],r['candidate']))
        result=dict(id=new_id('selection'),dataset='cn7',evaluation_id=evaluation_id,
            status='qualified_candidate' if eligible else 'selection_held',
            selected=eligible[0]['candidate'] if eligible else None,
            reports=reports,policy=self.policy,
            note='배포하지 않음. 비교 자료는 선택에 사용됐으므로 독립 최종 성능으로 주장하지 않음.')
        write(self.state/'selections'/f"{result['id']}.json",result)
        return result
