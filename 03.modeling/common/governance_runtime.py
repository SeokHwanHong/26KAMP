"""Approved operating policy: inspection budgets, reference governance and accountable CT/CD."""
import copy
import hashlib
from pathlib import Path
import math
import json
import pandas as pd
import numpy as np
from sklearn.base import clone
from workflow_runtime import Operations as WorkflowOperations
from decision_runtime import budget_flags,validate_product_ids
from pipeline_runtime import (read,write,digest,now,new_id,clean_id,feature_columns,fingerprints,
                              preprocess,data,fit_supervised,fit_oneclass,score,bin_spec,tv)


def timestamp(value):
    result=pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:raise ValueError('실제 생산 시각과 명시적 시간대 필요')
    return result.tz_convert('UTC')


class Operations(WorkflowOperations):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.dataset=='cn7' and (self.policy['cn7_decision_mode']!='budget' or self.policy['cn7_tie_rule']!='exact_k'):
            raise ValueError('공식 CN7 운영은 budget/exact_k. 기타 비교는 연구 경로 사용')
        cfg=self.governance_policy()
        self.policy.update(governance_version=cfg['id'],rg3_sampling=cfg.get('rg3_sampling'),
                           cn7_model_kind='rf',cn7_tie_rule='exact_k')
        self._sync_inspection_role()

    def _sync_inspection_role(self):
        if self.dataset=='cn7' and not self._advice_context:
            role=self.registry().get('inspection_role')
            if role:
                if self.registry()['active_models'].get(role['kind'])!=role['version']:raise ValueError('검사 역할/활성 모델 버전 불일치')
                self.policy.update(cn7_model_kind=role['kind'],inspection_fraction=role['fraction'],cn7_tie_rule='exact_k')

    def _ready(self):
        super()._ready();self._sync_inspection_role()

    def _inspection_version(self):
        kind=self.policy['cn7_model_kind']
        return self.active().get(kind)

    def governance_policy(self):
        pointer=read(self.state/'governance/policy_pointer.json')
        if not pointer:return dict(id='unconfigured',approvers={},training_selection={'mode':'cumulative'},rg3_sampling=None)
        folder=self.state/'governance/policies'/clean_id(pointer['id'])
        cfg=read(folder/'policy.json')
        if digest(folder/'policy.json')!=read(folder/'integrity.json')['sha256']:raise ValueError('운영 정책 변경')
        return cfg

    def governance_status(self):
        cfg=self.governance_policy()
        return dict(version=cfg['id'],approval_roles=list(cfg.get('approvers',{})),
                    sampling_configured=cfg.get('rg3_sampling') is not None,training_selection=cfg['training_selection'],
                    inspection_role=self.registry().get('inspection_role') if self.dataset=='cn7' else None,
                    note='미설정 승인/검사 작업은 보류. 기존 모델·기준선은 자동 교체하지 않음')

    def reference_status(self):
        pointer=read(self.state/'baseline_pointer.json')
        if not pointer:return dict(status='unprepared')
        _,m,_=self._verified_baseline(pointer['id'])
        return dict(baseline_id=m['id'],versions=m['versions'],source=m.get('reference_source',{'type':'initial_development'}),
                    reference_rows=m['reference_rows'],reference_patterns=m['reference_patterns'])

    def configure_governance(self,settings,actor,reason):
        if not isinstance(actor,str) or not actor.strip() or not reason.strip():raise ValueError('정책 승인자·사유 필요')
        allowed={'approvers','rg3_sampling','training_selection'}
        if set(settings)-allowed:raise ValueError('지원하지 않는 운영 설정')
        cfg=copy.deepcopy(settings)
        for action,actors in cfg.get('approvers',{}).items():
            if action not in ('ct','promote','reference','rollback') or not isinstance(actors,list) or not actors \
                    or any(not isinstance(a,str) or not a.strip() for a in actors):raise ValueError('승인 역할/담당자 설정 오류')
        sampling=cfg.get('rg3_sampling')
        if sampling is not None:
            if set(sampling)!=set(('waiting','normal','watch','review')):raise ValueError('RG3 단계별 검사 설정 필요')
            for item in sampling.values():
                if set(item)!=set(('fraction','random_share')) or not 0<float(item['fraction'])<=1 or not 0<float(item['random_share'])<1:
                    raise ValueError('전체 검사 비율 (0,1], 무작위 검사 비중 (0,1) 필요')
        selection=cfg.setdefault('training_selection',{'mode':'cumulative'})
        if selection.get('mode') not in ('cumulative','recent'):raise ValueError('학습 범위 설정 오류')
        if selection.get('max_patterns') is not None and (not isinstance(selection['max_patterns'],int) or selection['max_patterns']<1):raise ValueError('학습 패턴 상한 오류')
        if selection['mode']=='recent':
            if timestamp(selection['start'])>=timestamp(selection['end']):raise ValueError('학습 기간 범위 오류')
            if not isinstance(selection.get('process_version'),str) or not selection['process_version'].strip():raise ValueError('공정 버전 필요')
            if not isinstance(selection.get('max_patterns'),int) or selection['max_patterns']<1:raise ValueError('학습 고유 패턴 상한 필요')
        with self.operation():
            self._ready();cfg.update(id=new_id('governance'),dataset=self.dataset,actor=actor,reason=reason,created_at=now())
            folder=self.state/'governance/policies'/cfg['id'];write(folder/'policy.json',cfg)
            write(folder/'integrity.json',dict(sha256=digest(folder/'policy.json')))
            write(self.state/'governance/policy_pointer.json',dict(id=cfg['id']))
            self.policy.update(governance_version=cfg['id'],rg3_sampling=sampling)
            return cfg['id']

    def _binding(self,payload):
        pointer=read(self.state/'baseline_pointer.json')
        bound=dict(versions=super().active(),baseline=pointer,policy=self.governance_policy()['id'],
                   operating_policy=copy.deepcopy(self.policy),files={})
        if pointer:
            for name in ('manifest.json','reference_products.csv','reference_patterns.csv'):
                path=self.state/'baselines'/clean_id(pointer['id'])/name
                bound['files'][str(path.relative_to(self.state))]=digest(path)
        for version in bound['versions'].values():
            for name in ('manifest.json','model.joblib'):
                path=self.state/'models'/clean_id(version)/name
                bound['files'][str(path.relative_to(self.state))]=digest(path)
        for bid in payload.get('batch_ids',[]):
            folder=self.state/'batches'/clean_id(bid)
            for name in ('manifest.json','products.csv','patterns.csv','production_context.json'):
                path=folder/name
                if path.exists():bound['files'][str(path.relative_to(self.state))]=digest(path)
        for field,sub,name in [('drift_id','drift',None),('assessment_id','assessments','assessment.json'),
                               ('product_validation_id','governance/product_validations','report.json')]:
            if payload.get(field):
                ident=clean_id(payload[field]);path=self.state/sub/(ident+'.json' if name is None else ident+'/'+name)
                bound['files'][str(path.relative_to(self.state))]=digest(path)
        return bound

    def approve_action(self,action,payload,evidence,actor):
        with self.operation():
            self._ready();cfg=self.governance_policy()
            if actor not in cfg.get('approvers',{}).get(action,[]):raise ValueError('해당 작업의 승인 담당자 미설정/불일치')
            required=('normal_process_confirmed','input_error_excluded','changed_representations','label_source','evaluation_plan')
            if action in ('ct','reference'):
                if any(not evidence.get(k) for k in required) or evidence['normal_process_confirmed'] is not True or evidence['input_error_excluded'] is not True:
                    raise ValueError('변화·입력 오류 확인·정상 공정 검사·라벨 출처·평가 계획 필요')
            elif not evidence.get('reason'):raise ValueError('승인 사유 필요')
            record=dict(id=new_id('approval'),action=action,payload=copy.deepcopy(payload),evidence=copy.deepcopy(evidence),actor=actor,
                        binding=self._binding(payload),created_at=now())
            folder=self.state/'governance/approvals'/record['id'];write(folder/'approval.json',record)
            write(folder/'integrity.json',dict(sha256=digest(folder/'approval.json')))
            return record['id']

    def _approval(self,aid,action,payload):
        if not aid:raise ValueError('승인 기록 필요')
        folder=self.state/'governance/approvals'/clean_id(aid);r=read(folder/'approval.json')
        if digest(folder/'approval.json')!=read(folder/'integrity.json')['sha256']:raise ValueError('승인 기록 변경')
        if r['action']!=action or r['payload']!=payload or r['binding']!=self._binding(payload):raise ValueError('승인 대상/모델/기준선/정책/자료 변경: 재승인 필요')
        if r['actor'] not in self.governance_policy().get('approvers',{}).get(action,[]):raise ValueError('현재 승인 담당자 불일치')
        return r

    def ingest(self,frame,*args,production_time_column=None,process_version=None,metadata_source=None,**kwargs):
        times=None
        if production_time_column is not None:
            if production_time_column not in frame or not process_version or not metadata_source:raise ValueError('생산 시각 열·공정 버전·출처 필요')
            if not any(c in frame for c in ('record_id','product_id')):raise ValueError('생산 시각 연결에는 원천 제품 ID 필요')
            times=[timestamp(v).isoformat() for v in frame[production_time_column]]
        with self.operation():
            result=super().ingest(frame,*args,**kwargs)
            if times is not None and result.get('status')=='accepted':
                folder=self.state/'batches'/result['id'];p=pd.read_csv(folder/'products.csv',dtype={'record_id':str})
                id_column=next(c for c in ('record_id','product_id') if c in frame)
                context=dict(batch_id=result['id'],products_sha256=digest(folder/'products.csv'),process_version=process_version,
                             source=metadata_source,times=dict(zip(frame[id_column].astype(str),times)))
                path=folder/'production_context.json'
                if path.exists() and read(path)!=context:raise ValueError('최초 생산 메타데이터와 다른 재전송')
                if not path.exists():write(path,context)
            return result

    def _rg3_advice(self,result):
        answer=super()._rg3_advice(result);folder=self.state/'batches'/result['id']
        settings=self.policy.get('rg3_sampling');level=(result.get('drift') or {}).get('status','waiting')
        plan=dict(status='held_configuration',reason='단계별 검사 비율/무작위 비중 미설정',stage=level,
                  governance_version=self.policy.get('governance_version'),meaning='불확실성 안내와 실제 검사 대상은 별도')
        if settings:
            rows=read(folder/'inspection_advice.json')['records'];n=len(rows);item=settings[level]
            k=min(n,math.ceil(n*float(item['fraction'])))
            if k<2:plan.update(status='held_budget',reason='우선·무작위 검사 병행에 최소 2개 검사 예산 필요')
            else:
                ordered=sorted(rows,key=lambda r:({'out_of_reference':0,'insufficient_information':1}.get(r['evidence_level'],2),r['record_id']))
                random_k=min(k-1,max(1,math.ceil(k*float(item['random_share']))));priority_k=k-random_k
                priority=[r['record_id'] for r in ordered[:priority_k]];remaining=sorted(r['record_id'] for r in rows if r['record_id'] not in priority)
                seed=int(hashlib.sha256((result['id']+str(plan['governance_version'])).encode()).hexdigest()[:16],16)
                random_ids=np.random.default_rng(seed).choice(remaining,size=random_k,replace=False).tolist()
                plan.update(status='planned',total=n,budget_k=k,priority_ids=priority,random_ids=random_ids,
                            selected_ids=priority+random_ids,seed=seed,settings=item)
        write(folder/'inspection_plan.json',plan);answer['inspection_plan']=plan
        advice=read(folder/'inspection_advice.json');priority=set(plan.get('priority_ids',[]));random_ids=set(plan.get('random_ids',[]))
        advice['inspection_plan']=plan
        for row in advice['records']:
            row['inspection_planned']=row['record_id'] in priority or row['record_id'] in random_ids
            row['inspection_method']='priority' if row['record_id'] in priority else 'random' if row['record_id'] in random_ids else 'not_selected' if plan['status']=='planned' else 'held'
        write(folder/'inspection_advice.json',advice)
        table=pd.read_csv(folder/'inspection_advice.csv',dtype={'record_id':str})
        methods={r['record_id']:r['inspection_method'] for r in advice['records']}
        table['inspection_method']=table.record_id.map(methods);table['inspection_planned']=table.inspection_method.isin(['priority','random'])
        table.to_csv(folder/'inspection_advice.csv',index=False,encoding='utf-8-sig')
        text=folder/'inspection_advice.txt';text.write_text('불확실성 권고와 실제 검사 선정은 별도입니다. 실제 검사 계획: '+plan['status']+' / 선정 '+str(len(priority|random_ids))+'건\n'+text.read_text(encoding='utf-8'),encoding='utf-8')
        answer['inspection_advice']['planned_count']=len(priority|random_ids)
        manifest=read(folder/'decision_manifest.json')
        for name in ['inspection_plan.json','inspection_advice.json','inspection_advice.csv','inspection_advice.txt']:manifest['files'][name]=digest(folder/name)
        write(folder/'decision_manifest.json',manifest)
        return answer

    def _risk_corpus(self):
        x,y,_,dev,_,_=data(self.dataset);parts=[x.loc[dev].assign(label=y.loc[dev].to_numpy())]
        for bid in read(self.state/'batch_ledger.json',{'batches':[]})['batches']:
            m,_,p=self.batch(bid)
            if m['labeled']:parts.append(p[feature_columns()+['label']])
        for f in (self.state/'models').glob('*/manifest.json'):
            b,_=self.load_model(f.parent.name)
            if b.get('corpus') is not None:parts.append(b['corpus'][feature_columns()+['label']])
        return pd.concat(parts).groupby(feature_columns(),sort=False,dropna=False).label.max().reset_index()

    def _recent_patterns(self,batch_ids,selection):
        rows=[];start=timestamp(selection['start']);end=timestamp(selection['end'])
        for bid in batch_ids:
            m,p,context=self._batch_with_context(bid)
            if context['process_version']!=selection['process_version']:continue
            ids=p.record_id.astype(str);times=ids.map(context['times'])
            parsed=pd.to_datetime(times,utc=True,errors='raise');keep=(parsed>=start)&(parsed<end)
            rows.append(p.loc[keep,feature_columns()+['label']])
        if not rows or sum(map(len,rows))==0:raise ValueError('기간/공정 버전 학습 자료 없음')
        p=pd.concat(rows).groupby(feature_columns(),sort=False,dropna=False).label.max().reset_index()
        if len(p)>selection['max_patterns']:raise ValueError('학습 고유 패턴 상한 초과: 범위 재선정 필요')
        return p

    def _batch_with_context(self,bid):
        bid=clean_id(bid);m,_,_=self.batch(bid);folder=self.state/'batches'/bid
        p=pd.read_csv(folder/'products.csv',dtype={'record_id':str},float_precision='round_trip');context=read(folder/'production_context.json')
        if not m['labeled'] or not m['label_source'] or not context:raise ValueError('실제 라벨·출처·생산 메타데이터 필요')
        if context['products_sha256']!=digest(folder/'products.csv') or set(context['times'])!=set(p.record_id):raise ValueError('생산 메타데이터 제품 연결 오류')
        if not context.get('source') or not context.get('process_version'):raise ValueError('생산 시각 출처/공정 버전 필요')
        for value in context['times'].values():timestamp(value)
        return m,p,context

    def retrain(self,kind,batch_ids,drift_id,cause,base_version=None,*,approval_id=None):
        with self.operation():
            self._ready();payload=dict(kind=kind,batch_ids=list(batch_ids),drift_id=drift_id,cause=cause,base_version=base_version)
            self._approval(approval_id,'ct',payload);selection=self.governance_policy()['training_selection']
            if not cause.strip() or kind not in ('if','ocsvm','lr','rf') or not batch_ids or len(set(batch_ids))!=len(batch_ids):raise ValueError('CT 사유/모델/배치 목록 오류')
            drift=read(self.state/'drift'/f'{clean_id(drift_id)}.json');version=base_version or self.active().get(kind)
            if drift['status']!='review' or drift['versions']!=self.active() or not set(batch_ids)&set(drift['batch_ids']):raise ValueError('현재 모델의 검토 배치 필요')
            old,_=self.load_model(version)
            if old['kind']!=kind or version!=self.active().get(kind):raise ValueError('현재 CT 모델 불일치')
            archive=self._risk_corpus();archive_id=new_id('risk-history');folder=self.state/'governance/risk_history'/archive_id
            if selection['mode']=='cumulative':
                x,y,_,dev,_,_=data(self.dataset)
                parts=[old.get('corpus',x.loc[dev].assign(label=y.loc[dev].to_numpy()))[feature_columns()+['label']]]
                for bid in batch_ids:
                    m,_,p=self.batch(bid)
                    if not m['labeled'] or not m['label_source']:raise ValueError('실제 라벨·출처 필요')
                    parts.append(p[feature_columns()+['label']])
                selected=pd.concat(parts).groupby(feature_columns(),sort=False,dropna=False).label.max().reset_index()
            else:
                selected=self._recent_patterns(batch_ids,selection)
            known=dict(zip(fingerprints(archive[feature_columns()]),archive.label))
            selected['label']=[max(int(v),int(known.get(fp,0))) for fp,v in zip(fingerprints(selected[feature_columns()]),selected.label)]
            if selection.get('max_patterns') is not None and len(selected)>selection['max_patterns']:raise ValueError('학습 고유 패턴 상한 초과')
            if set(fingerprints(selected[feature_columns()]))&self.protected():raise ValueError('학습/평가 패턴 중복')
            xx=selected[feature_columns()];yy=selected.label;n0=int((yy==0).sum());n1=int((yy==1).sum())
            if n0<self.policy['min_train_normal'] or (kind in ('lr','rf') and n1<self.policy['min_train_risk']):raise ValueError('선택 학습 정상/위험 표본 부족')
            if kind in ('lr','rf'):
                b=fit_supervised(self.dataset,xx,yy,kind=kind,scenario=old['scenario'],C=old.get('C',1.),class_weight=old.get('class_weight'))
                if b.get('domain'):
                    from pipeline_runtime import library
                    values=library(self.dataset)['transform_features'](b['domain'],xx)
                else:values=xx
                b['model']=clone(old['model']).fit(values,yy);b['threshold']=old['threshold']
            else:b=fit_oneclass(self.dataset,xx.loc[yy==0],kind,old)
            folder.mkdir(parents=True,exist_ok=True);archive.to_csv(folder/'history.csv',index=False)
            write(folder/'manifest.json',dict(id=archive_id,sha256=digest(folder/'history.csv'),approval_id=approval_id,note='영구 위험 이력; 학습 범위와 별도'))
            v=self.save_candidate(b,xx,yy,source=dict(drift_id=drift_id,batch_ids=batch_ids,cause=cause,approval_id=approval_id,training_selection=selection,risk_history_id=archive_id),parent=version)
            write(self.state/'governance/training_records'/f'{v}.json',dict(version=v,approval_id=approval_id,selection=selection,risk_history_id=archive_id))
            return v

    def record_product_validation(self,assessment_id,frame,label_source):
        with self.operation():
            self._ready();a=self._validate_assessment(assessment_id);validate_product_ids(frame)
            if not any(c in frame for c in ('record_id','product_id')):raise ValueError('독립 제품 검사의 원천 제품 ID 필요')
            products,patterns=preprocess(frame,new_id('product-test'),label_source)
            if 'label' not in products:raise ValueError('실제 제품 검사 정답 필요')
            reference=pd.read_csv(self.state/'evaluations'/a['evaluation_id']/'patterns.csv',float_precision='round_trip')
            target=dict(zip(reference.fingerprint,reference.label))
            if any(fp not in target or int(target[fp])!=int(y) for fp,y in zip(patterns.fingerprint,patterns.label)):raise ValueError('독립 평가 패턴/라벨과 제품 검사 연결 불일치')
            current=self._inspection_version() if self.dataset=='cn7' else a['current']
            if not current:raise ValueError('현재 검사 기준 모델 필요')
            candidate=self.load_model(a['candidate'])[0];incumbent=self.load_model(current)[0]
            scores=score(candidate,products[feature_columns()]);old_scores=score(incumbent,products[feature_columns()])
            args=dict(fraction=self.policy['inspection_fraction'],ids=products.record_id,rule='exact_k',fingerprints=products.fingerprint)
            flags,_,k,_,_=budget_flags(scores,**args);old_flags,_,_,_,_=budget_flags(old_scores,**args)
            tp=int(products.loc[flags,'label'].sum());old_tp=int(products.loc[old_flags,'label'].sum());positives=int(products.label.sum())
            report=dict(id=new_id('product-validation'),assessment_id=assessment_id,candidate=a['candidate'],current_inspection_model=current,
                        evaluation_id=a['evaluation_id'],label_source=label_source,products=len(products),budget_k=k,positives=positives,
                        candidate_TP=tp,current_TP=old_tp,passed=positives>=self.policy['min_eval_risk'] and tp>old_tp,
                        policy_fraction=self.policy['inspection_fraction'],meaning='동일 제품 검사량에서 실제 제공 검사 정답 기준 발견 수')
            folder=self.state/'governance/product_validations'/report['id'];folder.mkdir(parents=True)
            products.to_csv(folder/'products.csv',index=False);report['products_sha256']=digest(folder/'products.csv')
            write(folder/'report.json',report);write(folder/'integrity.json',dict(sha256=digest(folder/'report.json')))
            return report['id']

    def promote(self,aid,*,approval_id=None,product_validation_id=None):
        with self.operation():
            self._ready()
            payload=dict(assessment_id=aid,product_validation_id=product_validation_id)
            self._approval(approval_id,'promote',payload)
            a=self._validate_assessment(aid)
            if self.dataset=='cn7' and a['kind'] in ('rf','lr','ocsvm'):
                if not product_validation_id:raise ValueError('CN7 독립 제품 검사 성과 필요')
                folder=self.state/'governance/product_validations'/clean_id(product_validation_id);p=read(folder/'report.json')
                if digest(folder/'report.json')!=read(folder/'integrity.json')['sha256'] or digest(folder/'products.csv')!=p['products_sha256']:raise ValueError('제품 검사 근거 변경')
                if not p['passed'] or p['assessment_id']!=aid or p['candidate']!=a['candidate'] or p['current_inspection_model']!=self._inspection_version() or p['policy_fraction']!=self.policy['inspection_fraction']:raise ValueError('제품 검사 성과/운영 정책 조건 미통과')
            r=copy.deepcopy(self.registry());kind=a['kind'];r['active_models'][kind]=a['candidate']
            if kind in ('if','ocsvm'):r['fixed_if' if kind=='if' else 'active_ocsvm']=a['candidate']
            if self.dataset=='cn7' and kind in ('rf','lr','ocsvm'):
                r['inspection_role']=dict(kind=kind,version=a['candidate'],fraction=self.policy['inspection_fraction'],tie_rule='exact_k',
                                          approval_id=approval_id,product_validation_id=product_validation_id,
                                          previous_role=r.get('inspection_role') or dict(kind='rf',version=self.active().get('rf'),fraction=self.policy['inspection_fraction'],tie_rule='exact_k'))
            r['history'].append(dict(action='promote',kind=kind,previous=a['current'],version=a['candidate'],assessment=aid,approval_id=approval_id,at=now()))
            result=self._transition(r,a['candidate']);self._sync_inspection_role();return result

    def rollback(self,kind,reason,*,approval_id=None):
        with self.operation():
            self._ready()
            self._approval(approval_id,'rollback',dict(kind=kind,reason=reason))
            r=copy.deepcopy(self.registry());current=r['active_models'].get(kind)
            history=[e for e in r['history'] if e['action']=='promote' and e['kind']==kind and e['version']==current]
            if not history:raise ValueError('롤백 이력 없음')
            old=history[-1]['previous'];self.load_model(old);r['active_models'][kind]=old
            if kind in ('if','ocsvm'):r['fixed_if' if kind=='if' else 'active_ocsvm']=old
            role=r.get('inspection_role')
            if role and role['kind']==kind:
                previous=role['previous_role'];r['inspection_role']=copy.deepcopy(previous)
                if previous['kind']==kind:r['inspection_role']['version']=old
            r['history'].append(dict(action='rollback',kind=kind,previous=current,version=old,reason=reason,approval_id=approval_id,at=now()))
            result=self._transition(r,old);self._sync_inspection_role();return result

    def resume_transition(self):
        result=super().resume_transition();self._sync_inspection_role();return result

    def _baseline_from(self,reference,patterns,versions,source,policy=None):
        bid=new_id('baseline');folder=self.state/'baselines'/bid;folder.mkdir(parents=True)
        reference.to_csv(folder/'reference_products.csv',index=False);patterns.to_csv(folder/'reference_patterns.csv',index=False)
        specs={c:bin_spec(reference[c]) for c in reference}
        for kind,v in versions.items():specs['score::'+kind]=bin_spec(score(self.load_model(v)[0],reference))
        write(folder/'manifest.json',dict(id=bid,dataset=self.dataset,created_at=now(),versions=versions,specs=specs,policy=policy or self.policy,
              reference_rows=len(reference),reference_patterns=len(patterns),reference_sha256=digest(folder/'reference_products.csv'),
              patterns_sha256=digest(folder/'reference_patterns.csv'),reference_source=source))
        self._verified_baseline(bid,versions);return bid

    def _build_baseline(self,versions):
        pointer=read(self.state/'baseline_pointer.json')
        if not pointer:return super()._build_baseline(versions)
        folder,m,reference=self._verified_baseline(pointer['id'])
        patterns=pd.read_csv(folder/'reference_patterns.csv',float_precision='round_trip')
        source=copy.deepcopy(m.get('reference_source',{}))
        source.update(reference_id=source.get('reference_id',m['id']),inherited_baseline=m['id'],note='승인된 입력 참조 보존; 모델 출력만 재계산')
        target_policy=copy.deepcopy(self.policy);pending=read(self.state/'transition_pending.json')
        if pending and pending['stage']!='completed':
            role=pending['registry'].get('inspection_role')
            if role:target_policy.update(cn7_model_kind=role['kind'],inspection_fraction=role['fraction'],cn7_tie_rule='exact_k')
        return self._baseline_from(reference,patterns,versions,source,target_policy)

    def create_baseline(self):
        with self.operation():
            self._ready();pointer=read(self.state/'baseline_pointer.json')
            if pointer:
                self._verified_baseline(pointer['id'],self.active())
                return pointer['id']
            return super().create_baseline()

    def update_reference(self,batch_ids,reason,*,approval_id=None):
        with self.operation():
            self._ready();payload=dict(batch_ids=list(batch_ids),reason=reason)
            self._approval(approval_id,'reference',payload);oldfolder,oldm,oldref=self.baseline();rows=[];processes=set();times=[]
            for bid in batch_ids:
                m,p,context=self._batch_with_context(bid)
                if not (p.label==0).all():raise ValueError('실제 정상 검사 라벨 필요')
                processes.add(context['process_version']);times.extend(timestamp(v) for v in context['times'].values())
                if set(p.fingerprint)&self.protected():raise ValueError('평가 자료의 참조 편입 금지')
                rows.append(p[feature_columns()])
            if len(processes)!=1:raise ValueError('새 정상 참조는 같은 공정 버전의 자료 필요')
            reference=pd.concat(rows,ignore_index=True);unique=reference.drop_duplicates()
            risk=self._risk_corpus();known_risk=set(fingerprints(risk.loc[risk.label==1,feature_columns()]))
            if set(fingerprints(unique))&known_risk:raise ValueError('과거 위험 패턴을 새 정상으로 편입하지 않음')
            if len(unique)<self.policy['min_train_normal']:raise ValueError('새 정상 참조 고유 패턴 부족')
            patterns=pd.concat([unique.assign(label=0),risk.loc[risk.label==1]],ignore_index=True)
            bid=self._baseline_from(reference,patterns,self.active(),dict(approval_id=approval_id,batch_ids=batch_ids,previous=oldm['id'],reason=reason,
                                                                        process_version=next(iter(processes)),production_start=min(times).isoformat(),production_end=max(times).isoformat()))
            newfolder,newm,_=self._verified_baseline(bid);changes={c:tv(self._represent(oldref,oldm)[c],self._represent(reference,oldm)[c]) for c in self._represent(oldref,oldm)}
            write(newfolder/'reference_comparison.json',dict(previous=oldm['id'],new=bid,changes=changes,approval_id=approval_id))
            journal=dict(id=new_id('transition'),stage='prepared',registry=self.registry(),version='reference:'+bid,
                         previous_registry=self.registry(),previous_baseline=read(self.state/'baseline_pointer.json'),baseline=bid,created_at=now())
            write(self.state/'transition_pending.json',journal);return self._complete_transition(journal)

    def rollback_reference(self,reason,*,approval_id=None):
        with self.operation():
            self._ready();pointer=read(self.state/'baseline_pointer.json');_,m,_=self._verified_baseline(pointer['id'])
            previous=m.get('reference_source',{}).get('previous')
            if not previous:raise ValueError('참조 롤백 근거 없음')
            self._approval(approval_id,'reference',dict(batch_ids=[],reason=reason,rollback_baseline=previous))
            folder,old,ref=self._verified_baseline(previous);patterns=pd.read_csv(folder/'reference_patterns.csv',float_precision='round_trip')
            bid=self._baseline_from(ref,patterns,self.active(),dict(previous=m['id'],rollback_of=previous,reason=reason,approval_id=approval_id))
            journal=dict(id=new_id('transition'),stage='prepared',registry=self.registry(),version='reference:'+bid,
                         previous_registry=self.registry(),previous_baseline=pointer,baseline=bid,created_at=now())
            write(self.state/'transition_pending.json',journal);return self._complete_transition(journal)
