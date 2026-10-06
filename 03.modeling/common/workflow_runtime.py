"""Version-pinned orchestration, recoverable transitions and evaluation lineage."""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import pandas as pd
from decision_runtime import Operations as DecisionOperations, LockHeld, LOCK_FILES
from pipeline_runtime import (read,write,digest,now,new_id,clean_id,feature_columns,
    fingerprints,data,ROOT,bin_spec,score,preprocess)
import numpy as np

LOCK_FILES.setdefault('operation','.operation.lock')


class Operations(DecisionOperations):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._operation_depth=0
        self._operation_owner=None
        self._advice_context=None

    @contextmanager
    def operation(self):
        if self._operation_depth and self._operation_owner==threading.get_ident():
            self._operation_depth+=1
            try:yield
            finally:self._operation_depth-=1
            return
        path=self.state/'.operation.lock'
        try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:raise LockHeld('운영 작업 진행 중: ingest/detect/모델 전환 종료 확인 필요') from None
        try:
            os.write(fd,json.dumps(dict(pid=os.getpid(),at=now())).encode())
            self._operation_depth=1
            self._operation_owner=threading.get_ident()
            yield
        finally:
            self._operation_depth=0;self._operation_owner=None;os.close(fd);path.unlink(missing_ok=True)

    def _ready(self):
        transition=read(self.state/'transition_pending.json')
        if transition and transition['stage']!='completed':
            raise ValueError('모델/기준선 전환 미완료: resume-transition 실행 필요')

    def active(self):
        if self._advice_context and self._operation_owner==threading.get_ident():return self._advice_context['versions']
        return super().active()

    def _verified_baseline(self,bid,versions=None):
        folder=self.state/'baselines'/clean_id(bid);m=read(folder/'manifest.json')
        if not m:raise ValueError('기준선 manifest 없음')
        for name,key in [('reference_products.csv','reference_sha256'),('reference_patterns.csv','patterns_sha256')]:
            if not (folder/name).is_file() or digest(folder/name)!=m[key]:raise ValueError('기준선 자료 변경/누락')
        if versions is not None and m['versions']!=versions:raise ValueError('모델/기준선 버전 불일치')
        for v in m['versions'].values():self.load_model(v)
        return folder,m,pd.read_csv(folder/'reference_products.csv',float_precision='round_trip')

    def baseline(self):
        if self._advice_context:return self._verified_baseline(self._advice_context['baseline'])
        self._ready()
        pointer=read(self.state/'baseline_pointer.json')
        if not pointer:
            self.create_baseline();pointer=read(self.state/'baseline_pointer.json')
        return self._verified_baseline(pointer['id'],super().active())

    def baseline_status(self):
        pending=read(self.state/'transition_pending.json')
        if pending and pending['stage']!='completed':return dict(valid=False,reasons=['모델 전환 미완료'],transition=pending)
        result=super().baseline_status()
        if result['valid']:
            try:self._verified_baseline(result['baseline'],super().active())
            except (ValueError,OSError) as exc:result.update(valid=False,reasons=[str(exc)])
        return result

    def initialize(self,*args,**kwargs):
        with self.operation():
            self._ready();return super().initialize(*args,**kwargs)

    def create_baseline(self):
        with self.operation():
            self._ready();return super().create_baseline()

    def ingest(self,*args,**kwargs):
        with self.operation():
            self._ready()
            pointer=read(self.state/'baseline_pointer.json')
            if pointer:self._verified_baseline(pointer['id'],super().active())
            return super().ingest(*args,**kwargs)

    def resume(self,batch_id):
        with self.operation():
            self._ready();return super().resume(batch_id)

    def reconcile_monitor(self):
        with self.operation():
            applied=super().reconcile_monitor()
            for path in sorted((self.state/'drift').glob('drift-*.json')):
                result=read(path)
                if result and result.get('id'):self._link_drift(result)
            return applied

    def detect(self):
        with self.operation():
            self._ready();return super().detect()

    def _run_detect(self):
        with self.operation():
            self._ready()
            result=super()._run_detect()
            pointer=read(self.state/'baseline_pointer.json')
            if pointer:result.setdefault('baseline',pointer['id'])
            result.setdefault('versions',super().active())
            result['policy_snapshot']=copy.deepcopy(self.policy)
            if result.get('id'):
                path=self.state/'drift'/f"{result['id']}.json"
                write(path,result)
                self._link_drift(result)
            return result

    def _link_drift(self,result):
        if 'policy_snapshot' not in result:
            _,m,_=self._verified_baseline(result['baseline'])
            result=dict(result,policy_snapshot=dict(m['policy'],drift_quantile=result['calibration']['quantile'],
                                                   bootstrap_repeats=result['calibration']['repeats']))
        links=read(self.state/'batch_drift_links.json',{})
        for bid in result['batch_ids']:
            prior=links.get(bid)
            if prior and prior['drift_id']!=result['id']:
                previous=read(self.state/'drift'/f"{prior['drift_id']}.json")
                if previous and previous['created_at']>result['created_at']:continue
            links[bid]=dict(drift_id=result['id'],status=result['status'],baseline=result['baseline'])
        write(self.state/'batch_drift_links.json',links)
        self._refresh_completed(result)

    def _drift_containing(self,bid):
        result=super()._drift_containing(bid)
        if result and 'policy_snapshot' not in result:
            _,m,_=self._verified_baseline(result['baseline'])
            result=dict(result,policy_snapshot=m['policy'])
        return result

    def _advise(self,result):
        path=self._processing_path(result['id']);proc=read(path)
        context=proc.get('advice_context')
        if not context:
            context=dict(versions=result['versions'],baseline=result['drift']['baseline'],
                policy=result['drift'].get('policy_snapshot',copy.deepcopy(self.policy)))
            proc['advice_context']=context;write(path,proc)
        old_policy=self.policy
        self._advice_context=context;self.policy=context['policy']
        try:
            for v in context['versions'].values():self.load_model(v)
            self._verified_baseline(context['baseline'])
            answer=super()._advise(result)
            folder=self.state/'batches'/result['id']
            revision=new_id('advice');history=folder/'advice_history'/revision;history.mkdir(parents=True)
            for p in folder.iterdir():
                if p.is_file() and p.name.startswith(('inspection_','distribution_guidance.','decision_manifest.')):shutil.copy2(p,history/p.name)
            meta=dict(revision=revision,created_at=now(),inference_versions=result['versions'],
                context=context,drift_id=result['drift'].get('id'),drift_status=result['drift']['status'])
            write(history/'context.json',meta);write(folder/'advice_latest.json',meta)
            answer['advice_revision']=revision
            return answer
        finally:self.policy=old_policy;self._advice_context=None

    def _refresh_completed(self,drift):
        for bid in drift['batch_ids']:
            path=self._processing_path(bid);proc=read(path)
            if not proc or proc['stage']!='completed' or proc.get('latest_drift_id')==drift['id']:continue
            latest=read(self.state/'batch_drift_links.json',{}).get(bid)
            if latest and latest['drift_id']!=drift['id']:continue
            if (proc.get('drift') or {}).get('status')!='waiting':continue
            original_context=proc.get('advice_context')
            proc['advice_context']=dict(versions=read(self.state/'batches'/bid/'manifest.json')['versions'],
                baseline=drift['baseline'],policy=drift['policy_snapshot'])
            write(path,proc)
            try:
                self._advise(dict(read(self.state/'batches'/bid/'manifest.json'),drift=drift))
                proc.update(latest_drift_id=drift['id'],latest_advice_error=None)
            except Exception as exc:proc['latest_advice_error']=f'{type(exc).__name__}: {exc}'
            finally:
                proc['advice_context']=original_context;write(path,proc)

    def batch_status(self,bid):
        bid=clean_id(bid)
        try:proc=read(self._processing_path(bid))
        except (ValueError,OSError) as exc:proc=dict(stage='evidence_error',error=str(exc))
        link=read(self.state/'batch_drift_links.json',{}).get(bid)
        return dict(processing=proc,latest_drift=link,
            distribution_final=bool(link),advice=read(self.state/'batches'/bid/'advice_latest.json'))

    def pending_batches(self):
        out=[]
        for folder in sorted((self.state/'batches').glob('*')):
            if not folder.is_dir():continue
            try:
                proc=read(folder/'processing.json');manifest=read(folder/'manifest.json',{})
                if proc and proc.get('stage')!='completed':out.append(dict(batch_id=folder.name,stage=proc['stage'],error=proc.get('error')))
                elif proc and proc.get('latest_advice_error'):out.append(dict(batch_id=folder.name,stage='latest_advice_failed',error=proc['latest_advice_error']))
                elif not proc and manifest.get('status')=='accepted':out.append(dict(batch_id=folder.name,stage='processing_evidence_missing'))
            except (ValueError,OSError,TypeError,AttributeError,KeyError) as exc:
                out.append(dict(batch_id=folder.name,stage='evidence_error',error=str(exc)))
        for bid in self._index()['reservations']:
            if not any(r['batch_id']==bid for r in out):out.append(dict(batch_id=bid,stage='reservation_pending'))
        return out

    def refresh_advice(self,bid):
        with self.operation():
            self._ready();status=self.batch_status(bid);link=status['latest_drift']
            if not link:raise ValueError('확정된 후속 감지 없음')
            drift=read(self.state/'drift'/f"{link['drift_id']}.json")
            self._refresh_completed(drift);return self.batch_status(bid)

    def _build_baseline(self,versions):
        x,y,_,dev,_,_=data(self.dataset)
        meta=pd.read_csv(ROOT/f'data/processed/{self.dataset}/conservative/labeled_metadata.csv')
        ref=x.loc[dev].iloc[np.repeat(np.arange(len(dev)),meta.loc[dev,'total_count'].to_numpy(int))].reset_index(drop=True)
        bid=new_id('baseline');folder=self.state/'baselines'/bid;folder.mkdir(parents=True)
        ref.to_csv(folder/'reference_products.csv',index=False)
        x.loc[dev].assign(label=y.loc[dev].to_numpy()).to_csv(folder/'reference_patterns.csv',index=False)
        specs={c:bin_spec(ref[c]) for c in ref}
        for kind,v in versions.items():specs['score::'+kind]=bin_spec(score(self.load_model(v)[0],ref))
        write(folder/'manifest.json',dict(id=bid,dataset=self.dataset,created_at=now(),versions=versions,
            specs=specs,policy=self.policy,reference_rows=len(ref),reference_patterns=len(dev),
            reference_sha256=digest(folder/'reference_products.csv'),patterns_sha256=digest(folder/'reference_patterns.csv'),
            note='development input reference preserved; staged model output baseline'))
        self._verified_baseline(bid,versions);return bid

    def _transition(self,registry,version):
        journal=dict(id=new_id('transition'),stage='preparing',registry=registry,version=version,
            previous_registry=self.registry(),previous_baseline=read(self.state/'baseline_pointer.json'),baseline=None,created_at=now())
        write(self.state/'transition_pending.json',journal)
        return self._complete_transition(journal)

    def _complete_transition(self,journal):
        try:
            if not journal['baseline']:
                journal['baseline']=self._build_baseline(journal['registry']['active_models'])
                journal['stage']='prepared';write(self.state/'transition_pending.json',journal)
            self._verified_baseline(journal['baseline'],journal['registry']['active_models'])
            for v in journal['registry']['active_models'].values():self.load_model(v)
            with self.lock():
                write(self.state/'registry.json',journal['registry'])
                write(self.state/'baseline_pointer.json',dict(id=journal['baseline']))
                journal.update(stage='completed',completed_at=now(),error=None)
                write(self.state/'transition_pending.json',journal)
            write(self.state/'transitions'/f"{journal['id']}.json",journal)
            return journal['version']
        except Exception as exc:
            journal['error']=f'{type(exc).__name__}: {exc}'
            write(self.state/'transition_pending.json',journal);raise

    def resume_transition(self):
        with self.operation():
            journal=read(self.state/'transition_pending.json')
            if not journal:raise ValueError('전환 기록 없음')
            if journal['stage']=='completed':return journal['version']
            return self._complete_transition(journal)

    def _validate_assessment(self,aid):
        folder=self.state/'assessments'/clean_id(aid);a=read(folder/'assessment.json');integrity=read(folder/'integrity.json')
        if not integrity or digest(folder/'assessment.json')!=integrity['assessment_sha256']:raise ValueError('평가 기록 변조')
        if not a['passed']:raise ValueError('배포 조건 미통과')
        if a.get('decision_mode')=='budget' and a['candidate_budget_metrics']['tie_rule']!='exact_k':raise ValueError('exact_k 필요')
        self._verify_evaluation_manifest(a['evaluation_id'])
        if a.get('evaluation_manifest_sha256')!=digest(self.state/'evaluations'/a['evaluation_id']/'manifest.json'):raise ValueError('평가 계약 변경')
        if self._selection_overlap(a['evaluation_id']):raise ValueError('선정 자료와 중복: 별도 최종 평가 필요')
        if digest(folder/'distribution_guidance.json')!=a['guidance_sha256']:raise ValueError('구간 안내 변경')
        if digest(self.state/'evaluations'/a['evaluation_id']/'patterns.csv')!=a['evaluation_sha256']:raise ValueError('평가 자료 변경')
        if self.active().get(a['kind'])!=a['current']:raise ValueError('운영 모델 변경: 재평가 필요')
        for field,hashfield in [('candidate','candidate_sha256'),('current','current_sha256')]:
            if self.load_model(a[field])[1]['artifact_sha256']!=a[hashfield]:raise ValueError('평가 후 모델 변경')
        return a

    def promote(self,aid):
        with self.operation():
            self._ready();a=self._validate_assessment(aid);r=copy.deepcopy(self.registry());kind=a['kind']
            r['active_models'][kind]=a['candidate']
            if kind in ('if','ocsvm'):r['fixed_if' if kind=='if' else 'active_ocsvm']=a['candidate']
            r['history'].append(dict(action='promote',kind=kind,previous=a['current'],version=a['candidate'],assessment=aid,at=now()))
            return self._transition(r,a['candidate'])

    def rollback(self,kind,reason):
        with self.operation():
            self._ready()
            if not reason.strip():raise ValueError('롤백 사유 필요')
            r=copy.deepcopy(self.registry());current=r['active_models'].get(kind)
            h=[e for e in r['history'] if e['action']=='promote' and e['kind']==kind and e['version']==current]
            if not h:raise ValueError('롤백 이력 없음')
            old=h[-1]['previous'];self.load_model(old);r['active_models'][kind]=old
            if kind in ('if','ocsvm'):r['fixed_if' if kind=='if' else 'active_ocsvm']=old
            r['history'].append(dict(action='rollback',kind=kind,previous=current,version=old,reason=reason,at=now()))
            return self._transition(r,old)

    def _evaluation_evidence(self,folder,allow_legacy=False):
        """Legacy evidence can restrict independence, but never acquires audit trust implicitly."""
        m=read(folder/'manifest.json')
        if not isinstance(m,dict) or m.get('id')!=folder.name:raise ValueError('평가 manifest 형식/ID 오류')
        integrity=read(folder/'manifest_integrity.json')
        if integrity:
            if digest(folder/'manifest.json')!=integrity['manifest_sha256']:raise ValueError('평가 manifest 변경')
        elif not allow_legacy:raise ValueError('구형 평가: migrate-evaluation으로 후속 평가 사본을 이관하세요')
        if digest(folder/'patterns.csv')!=m['sha256']:raise ValueError('평가 CSV 변경')
        p=pd.read_csv(folder/'patterns.csv',float_precision='round_trip')
        if m['dataset']!=self.dataset or m['purpose'] not in ('independent','historical_followup'):raise ValueError('평가 역할/데이터셋 오류')
        if 'label' not in p or not set(p.label.unique())<={0,1}:raise ValueError('평가 정답은 완전한 0/1이어야 함')
        fp=fingerprints(p[feature_columns()])
        if fp!=p.fingerprint.tolist() or fp!=m['fingerprints'] or len(set(fp))!=len(fp):raise ValueError('평가 fingerprint 정합성 오류')
        return m,p,bool(integrity)

    def _verify_evaluation_manifest(self,eid):
        folder=self.state/'evaluations'/clean_id(eid)
        self._evaluation_evidence(folder)
        return digest(folder/'manifest.json')

    def register_evaluation(self,frame,label_source,purpose='independent',evaluation_id=None,*,migration=None):
        with self.operation():
            if purpose not in ('independent','historical_followup'):raise ValueError('evaluation purpose')
            eid=clean_id(evaluation_id or new_id('eval'));folder=self.state/'evaluations'/eid
            if folder.exists():raise ValueError('이미 등록된 평가 ID')
            # Preflight prior evidence before creating any registered evaluation folder.
            prior=[]
            for f in sorted((self.state/'evaluations').glob('*/manifest.json')):
                m0,p0,verified=self._evaluation_evidence(f.parent,allow_legacy=True)
                prior.append((m0,p0,verified))
            _,p=preprocess(frame,eid,label_source)
            if 'label' not in p:raise ValueError('평가에는 실제 라벨 필요')
            trained=set()
            for f in (self.state/'models').glob('*/manifest.json'):
                trained.update(read(f).get('train_fingerprints',[]))
            x,_,_,dev,test,_=data(self.dataset);trained.update(fingerprints(x.loc[dev]))
            target=set(p.fingerprint)
            if target & trained:raise ValueError('학습 이력이 있는 패턴은 독립 평가 자료로 등록 불가')
            if target & set(fingerprints(x.loc[test])) or migration:purpose='historical_followup'
            content=sorted(zip(fingerprints(p[feature_columns()]),p.label.astype(int)))
            canonical=hashlib.sha256(json.dumps(content).encode()).hexdigest()
            previous=[];legacy_overlap=[]
            for m0,p0,verified in prior:
                old_content=sorted(zip(p0.fingerprint,p0.label.astype(int)))
                cc=hashlib.sha256(json.dumps(old_content).encode()).hexdigest()
                if cc==canonical:
                    previous.append(m0['id'])
                    if m0['purpose']=='historical_followup':purpose='historical_followup'
                if not verified and target & set(p0.fingerprint):
                    legacy_overlap.append(m0['id']);purpose='historical_followup'
            # A failed registration lives outside evaluations, so it cannot protect/poison new data.
            stage=self.state/'evaluation_staging'/eid
            contract=dict(id=eid,dataset=self.dataset,canonical_sha256=canonical,label_source=label_source,
                          purpose=purpose,migration=migration)
            with self.lock():
                stage.mkdir(parents=True,exist_ok=True)
                old_contract=read(stage/'registration_contract.json')
                if old_contract and old_contract!=contract:raise ValueError('등록 재개 자료/역할이 최초 요청과 다름')
                if not old_contract and any(stage.iterdir()):raise ValueError('등록 준비 근거 손상: 자동 재사용 불가')
                write(stage/'registration_contract.json',contract)
                p.to_csv(stage/'patterns.csv',index=False)
                m=dict(id=eid,dataset=self.dataset,purpose=purpose,label_source=label_source,
                       fingerprints=p.fingerprint.tolist(),created_at=now(),sha256=digest(stage/'patterns.csv'),
                       canonical_sha256=canonical,previous_evaluation_ids=previous,legacy_evaluation_ids=legacy_overlap,
                       migration=migration,legacy_note='구형 자료와 겹치면 후속 평가만 허용; 과거 기록은 수정하지 않음')
                write(stage/'manifest.json',m)
                write(stage/'manifest_integrity.json',dict(manifest_sha256=digest(stage/'manifest.json'),
                                                         patterns_sha256=m['sha256'],created_at=now()))
                self._evaluation_evidence(stage)
                folder.parent.mkdir(parents=True,exist_ok=True)
                os.replace(stage,folder)
            return eid

    def migrate_evaluation(self,evaluation_id,label_source,reason,new_evaluation_id=None):
        """Explicit immutable legacy copy. Historical status can never be upgraded to independent."""
        if not reason.strip() or not label_source.strip():raise ValueError('이관 사유와 라벨 출처 필요')
        with self.operation():
            folder=self.state/'evaluations'/clean_id(evaluation_id)
            m,p,verified=self._evaluation_evidence(folder,allow_legacy=True)
            if verified:raise ValueError('이미 무결성 기록이 있는 평가')
            migration=dict(source_evaluation_id=evaluation_id,source_manifest_sha256=digest(folder/'manifest.json'),
                           source_patterns_sha256=digest(folder/'patterns.csv'),original_purpose=m['purpose'],reason=reason,
                           policy='historical copy only; original evidence and independence are not retroactively certified')
            return self.register_evaluation(p[feature_columns()].assign(PassOrFail=p.label),label_source,
                                            'historical_followup',new_evaluation_id,migration=migration)

    def evaluation_registration_status(self):
        return [dict(evaluation_id=p.name,contract=read(p/'registration_contract.json'),
                     message='동일 입력·출처·purpose와 evaluation_id로 register-evaluation 재시도')
                for p in sorted((self.state/'evaluation_staging').glob('*')) if p.is_dir()]

    def _selection_overlap(self,eid):
        self._verify_evaluation_manifest(eid)
        target=set(read(self.state/'evaluations'/eid/'manifest.json')['fingerprints'])
        for selected in self.selection_evaluations():
            m,_,_=self._evaluation_evidence(self.state/'evaluations'/clean_id(selected),allow_legacy=True)
            if target & set(m['fingerprints']):return True
        return False

    def evaluate(self,*args,**kwargs):
        with self.operation():
            self._ready();return super().evaluate(*args,**kwargs)

    def select_cn7(self,*args,**kwargs):
        with self.operation():
            self._ready();return super().select_cn7(*args,**kwargs)

    def retrain(self,*args,**kwargs):
        with self.operation():
            self._ready();return super().retrain(*args,**kwargs)
