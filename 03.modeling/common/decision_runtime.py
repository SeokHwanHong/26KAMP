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

2026-10-05 team review revision (F01-F07; see Operations docstring)
- Batch dedup key independent of row order; explicit product IDs first, content as fallback.
- Ingest split into commit -> drift -> advice with processing.json; failed later stages resume, never re-store.
- Duplicate check + registration serialized by an ingest lock.
- Drift 'waiting' -> uncertainty 'pending'(판단 대기), not 'low'.
- One inspection tie rule (default exact_k) for batch flags and evaluation.
- Drift calibration cache tied to quantile/repeats/baseline/code; evaluation purpose record hashed.
"""
import math
import re
import hashlib
import json
import os
import shutil
import socket
import time
from contextlib import contextmanager
from pathlib import Path
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




UNCERTAINTY_TEXT={'high':'높음','medium':'중간','low':'낮음','pending':'판단 대기'}


def uncertainty_level(record_level,drift_status):
    """Uncertainty of judgement, NOT defect risk.

    pending(판단 대기): drift has not produced a judgement yet (waiting for samples). It must not be
    read as 'low'. Record-level evidence (out of range / information shortage) still applies.
    """
    if drift_status=='review' or record_level=='out_of_reference':return 'high'
    if drift_status=='watch' or record_level=='insufficient_information':return 'medium'
    if drift_status!='normal':return 'pending'
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


TIE_RULES=('exact_k','all_ties')


def _check_fraction(fraction):
    if not (0<fraction<=1):raise ValueError('검사 비율은 0 초과 1 이하여야 함(0%는 허용하지 않음: 최소 1개 검사)')


def budget_flags(scores,fraction,ids=None,rule='exact_k',fingerprints=None):
    """Top-k inspection flags for one batch (unit: product).

    Selection rule (shared with budget_metrics): higher score first, ties by fingerprint
    (input pattern) string order, then record_id. The order does not use labels.
    exact_k  : exactly k products.
    all_ties : every product tied at the cutoff is flagged; flagged may exceed k (recorded).
    """
    if rule not in TIE_RULES:raise ValueError(f'동점 규칙은 {TIE_RULES} 중 하나')
    _check_fraction(fraction)
    s=np.asarray(scores,float);n=len(s);k=min(n,max(1,int(math.ceil(n*fraction))))
    ids=np.asarray([str(i) for i in (ids if ids is not None else range(n))])
    fps=np.asarray([str(f) for f in (fingerprints if fingerprints is not None else ['']*n)])
    order=pd.DataFrame(dict(s=s,f=fps,i=ids,p=np.arange(n))).sort_values(
        ['s','f','i'],ascending=[False,True,True],kind='stable').p.to_numpy()
    cutoff=float(s[order[k-1]])
    if rule=='exact_k':
        flag=np.zeros(n,bool);flag[order[:k]]=True
    else:flag=s>=cutoff
    rank=pd.Series(-s).rank(method='min').astype(int).to_numpy()
    tied=int((s==cutoff).sum());at_or_above=int((s>=cutoff).sum())
    ties=dict(tie_rule=rule,tie_break='fingerprint, record_id' if rule=='exact_k' else None,
              tied_at_cutoff=tied,ties_cut=bool(rule=='exact_k' and at_or_above>k),over_budget=int(flag.sum())-k)
    return flag,rank,k,cutoff,ties


def budget_metrics(y,scores,fraction,rule='exact_k',fingerprints=None):
    """Evaluation counterpart of budget_flags (unit: unique pattern), SAME selection rule.

    found_TP: risks found by the patterns actually selected under the rule (used for decisions).
    random_tie_* : what random tie-breaking would give; reference only, not used for decisions.
    """
    if rule not in TIE_RULES:raise ValueError(f'동점 규칙은 {TIE_RULES} 중 하나')
    y=np.asarray(y).astype(int);s=np.asarray(scores,float)
    flag,_,k,cutoff,ties=budget_flags(s,fraction,None,rule,fingerprints)
    found=int(y[flag].sum());ref=topk(y,s,fraction)
    flagged=int(flag.sum())
    return dict(k=k,positives=int(y.sum()),flagged=flagged,inspected=flagged,over_budget=flagged-k,found_TP=found,
                reference_only=rule!='exact_k',
                tie_rule=rule,tie_break=ties['tie_break'],tied_at_cutoff=ties['tied_at_cutoff'],
                ties_cut=ties['ties_cut'],unit='unique_pattern',
                random_tie_expected_TP=ref['expected_TP'],random_tie_min_TP=ref['min_TP'],random_tie_max_TP=ref['max_TP'],
                note=('판정은 found_TP(실제 선택 규칙과 동일). random_tie_*는 동점을 무작위로 고를 때의 참고값. '
                      'all_ties는 실제 검사 수가 후보마다 달라 동일 검사량 비교가 아니므로 참고 전용'))


class ProductIdError(ValueError):
    """Explicit product ID column present but values are not valid production identifiers (contract A)."""


ID_COLUMNS=('record_id','product_id')


def validate_product_ids(frame):
    """Contract A: if record_id/product_id exists, every value must be a usable production ID.
    Returns the column used (record_id preferred, same as preprocess) or None (contract C)."""
    col=next((c for c in ID_COLUMNS if c in frame),None)
    if col is None:return None
    bad=[]
    for i,v in enumerate(frame[col].tolist()):
        if v is None or isinstance(v,(bool,np.bool_)) or (isinstance(v,float) and not math.isfinite(v)):
            bad.append((i,'결측'));continue
        if isinstance(v,(float,np.floating)):
            bad.append((i,'실수형(정수·문자 ID만 허용)'));continue
        if not isinstance(v,(str,int,np.integer)):
            bad.append((i,f'허용하지 않는 자료형 {type(v).__name__}'));continue
        text=str(v)
        if not text.strip():bad.append((i,'빈 값'))
        elif text!=text.strip():bad.append((i,'앞뒤 공백'))
    if bad:
        head=', '.join(f'{i}행 {why}' for i,why in bad[:5])
        raise ProductIdError(f'{col} 열이 있으나 유효한 생산 제품 ID가 아님({len(bad)}건: {head}). '
                             'ID 열을 고치거나, 생산 ID가 없는 자료면 ID 열 없이 보내세요(내용 기반 중복 의심 경로).')
    return col


class LockHeld(FileExistsError):
    """Lock already present; carries age and owner for the operator."""


LOCK_FILES={'writer':'.writer.lock','ingest':'.ingest.lock'}


class Operations(BaseOperations):
    """Decision layer.

    2026-10-05 revision (team review F01-F07):
    - F01 order-independent batch content key (pattern summary), explicit IDs are the primary evidence.
    - F02 ingest = commit(stored+ledger+dedup index) -> drift -> advice, tracked in processing.json.
          A failed later stage is resumed with the same batch_id (or resume); it is never re-stored.
    - F03 duplicate check and registration run inside one ingest lock.
    - F04 drift 'waiting' gives uncertainty 'pending'(판단 대기), never 'low'.
    - F05 one tie rule (default exact_k) shared by batch flags and evaluation.
    - F06 drift calibration cache is tied to quantile/repeats/baseline/runtime code; stale ones archived.
    - F07 evaluation manifest(purpose) hashed at registration and verified at evaluate/promote.
    pipeline_runtime.py is unchanged (LR audit hash).
    """
    def __init__(self,dataset,state_root=None,policy=None):
        super().__init__(dataset,state_root,policy)
        self.policy.setdefault('min_recall',.5)
        self.policy.setdefault('min_precision',.2)
        self.policy.setdefault('rg3_history_triggers_recheck',False)
        self.policy.setdefault('cn7_decision_mode','budget')      # budget | threshold
        self.policy.setdefault('cn7_model_kind','rf')
        self.policy.setdefault('cn7_tie_rule','exact_k')         # exact_k | all_ties
        self.policy.setdefault('min_budget_capture',.5)          # share of evaluation risks found within budget
        self.policy.setdefault('lock_stale_seconds',600)
        self.policy.setdefault('acceptance_policy_note','잠정 기준: 현장 검사 비용·누락 허용량 합의 필요')
        if self.policy['cn7_tie_rule'] not in TIE_RULES:raise ValueError(f'cn7_tie_rule은 {TIE_RULES} 중 하나')
        _check_fraction(self.policy['inspection_fraction'])
        self._defer_detect=False

    # ---- locks with owner/age (writer: short state writes, ingest: one whole batch registration) ----
    @contextmanager
    def _exclusive(self,which):
        path=self.state/LOCK_FILES[which]
        try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            age=time.time()-path.stat().st_mtime
            try:owner=path.read_text(encoding='utf-8') or '기록 없음'
            except OSError:owner='읽기 실패'
            raise LockHeld(f'{which} 잠금 사용 중({age:.0f}초 전 생성, 소유: {owner}). 실행 중인 작업이 없으면 '
                           f'pipeline_cli.py --dataset {self.dataset} unlock --lock {which} --reason <사유> 로 해제하세요: {path}') from None
        try:
            os.write(fd,json.dumps(dict(pid=os.getpid(),host=socket.gethostname(),at=now()),ensure_ascii=False).encode('utf-8'))
            yield
        finally:
            os.close(fd)
            try:path.unlink()
            except FileNotFoundError:pass

    @contextmanager
    def lock(self):
        with self._exclusive('writer'):yield

    def clear_lock(self,reason,force=False,which='writer'):
        """Remove a stale lock after an explicit operator decision; journaled."""
        if not reason or not reason.strip():raise ValueError('잠금 해제 사유 필요')
        if which not in LOCK_FILES:raise ValueError(f'잠금 종류는 {tuple(LOCK_FILES)} 중 하나')
        path=self.state/LOCK_FILES[which]
        if not path.exists():return dict(status='no_lock',lock=which)
        age=time.time()-path.stat().st_mtime;owner=path.read_text(encoding='utf-8',errors='replace')
        if age<self.policy['lock_stale_seconds'] and not force:
            raise ValueError(f'잠금 생성 후 {age:.0f}초: 작업이 진행 중일 수 있음. 확인 후 --force 사용')
        path.unlink()
        history=read(self.state/'lock_history.json',[]);entry=dict(at=now(),lock=which,reason=reason,age_seconds=age,owner=owner,force=force)
        history.append(entry);write(self.state/'lock_history.json',history)
        return dict(status='cleared',**entry)

    # ---- F01: duplicate batch key ----
    def _dedup_key(self,frame,batch_id,label_source):
        """Order-independent content key + explicit product IDs.

        Content = per-pattern summary (fingerprint, product/normal/defect counts, max label), so a
        re-sent file with reordered rows gives the same key. Row order and pandas row numbers are ignored.
        """
        validate_product_ids(frame)
        products,patterns=preprocess(frame,batch_id,label_source)
        cols=['fingerprint','product_count']+[c for c in ('label','normal_count','defect_count') if c in patterns]
        body=patterns[cols].sort_values('fingerprint',kind='stable').reset_index(drop=True)
        content=hashlib.sha256(body.to_csv(index=False,lineterminator='\n').encode('utf-8')).hexdigest()
        # Product ID contract (2026-10-05, A + C):
        #  A) record_id/product_id present -> production product ID, unique within the dataset (cn7/rg3),
        #     issued by the source system. Primary evidence for duplicates and for the product-input-label link.
        #  C) absent -> compatibility path: internal '<batch_id>:<row>' ids are NOT compared across batches;
        #     duplicates can only be suspected from content. 'Unnamed: 0' is a per-file row number, never an ID.
        explicit=any(c in frame for c in ID_COLUMNS)
        link=None
        if explicit:
            label=products['label'].astype(str) if 'label' in products else pd.Series(['NA']*len(products))
            rows=pd.DataFrame(dict(id=products.record_id.astype(str),fp=products.fingerprint,label=label.to_numpy()))
            rows=rows.sort_values(['id','fp','label'],kind='stable').reset_index(drop=True)
            link=hashlib.sha256(rows.to_csv(index=False,lineterminator='\n').encode('utf-8')).hexdigest()
        return content,(set(products.record_id) if explicit else set()),explicit,link

    def _held(self,batch_id,reason):
        folder=self.state/'batches'/batch_id
        with self.lock():folder.mkdir(parents=True,exist_ok=False)
        manifest=dict(id=batch_id,status='held',reason=reason,created_at=now())
        write(folder/'manifest.json',manifest);return manifest

    def _index(self):
        index=read(self.state/'dedup_index.json',{})
        for k in ('content','ids','reservations'):index.setdefault(k,{})
        return index

    def _duplicate_check(self,key,index,batch_id=None):
        """Return (held_reason or None, note or None). IDs first; content is the fallback evidence.
        Entries reserved by this same batch_id (an earlier interrupted attempt) are not duplicates."""
        content,ids,explicit=key[:3]
        same=index['content'].get(content)
        if same==batch_id:same=None
        reused=sorted(str(i) for i in ids if index['ids'].get(i) not in (None,batch_id))
        if reused:
            head='이미 수집한 배치와 동일한 내용, ' if same else ''
            return f"{head}이전 배치와 제품 ID 중복 {len(reused)}건: {index['ids'][reused[0]]}",None
        if same and not explicit:
            return (f'이미 수집한 배치와 동일한 내용(중복 의심): {same}. 제품 ID가 없어 내용으로만 판단했습니다. '
                    '실제로 다른 생산 건이면 record_id 또는 product_id 열을 넣어 다시 보내세요.'),None
        if same:
            return None,f'내용이 배치 {same}와 같지만 제품 ID가 모두 새로워 별도 생산 건으로 수집함'
        return None,None

    # ---- F02/F03: ingest = commit -> drift -> advice ----
    def _processing_path(self,batch_id):return self.state/'batches'/batch_id/'processing.json'

    def _reserve(self,batch_id,key,note):
        """Write the duplicate index entry BEFORE storing the batch.

        If any later write (files, ledger, processing state) fails, the same content / product IDs stay
        blocked under another batch_id, and pending_batches() reports the batch as incomplete."""
        with self.lock():
            index=self._index()
            index['content'].setdefault(key[0],batch_id)
            index['ids'].update({i:batch_id for i in key[1]})
            index['reservations'][batch_id]=dict(content_sha256=key[0],ids=sorted(map(str,key[1])),link_sha256=key[3],
                id_mode='explicit_product_id' if key[2] else 'no_product_id_compat',note=note,reserved_at=now())
            write(self.state/'dedup_index.json',index)

    def _release(self,batch_id):
        """Undo a reservation when the base ingest held the batch (nothing stored)."""
        with self.lock():
            index=self._index();r=index['reservations'].pop(batch_id,None)
            if r:
                if index['content'].get(r['content_sha256'])==batch_id:index['content'].pop(r['content_sha256'])
                for i in r['ids']:
                    if index['ids'].get(i)==batch_id:index['ids'].pop(i)
            write(self.state/'dedup_index.json',index)

    @staticmethod
    def _conflict(batch_id,reason):
        return dict(id=batch_id,status='conflict',reason=reason)

    LINK_CONFLICT=('제품 ID·입력·라벨 연결이 최초 수집과 다릅니다(제품 ID 교체, ID별 입력·라벨 변경, 라벨 정정 포함). '
                   '기존 배치는 그대로 두었습니다. 같은 자료의 재전송만 허용합니다. 새 생산은 새 batch_id로 보내고, '
                   '라벨 정정은 재전송으로 처리하지 않습니다(별도 정정 절차 필요, 현재 미구현).')

    def _reservation_conflict(self,batch_id,key):
        """Same batch_id re-sent while its reservation is still open: content and product IDs must match.
        Otherwise refuse and keep the reservation (the old products were never collected, the new ones
        must not inherit or overwrite their IDs)."""
        reservation=self._index()['reservations'].get(batch_id)
        if not reservation:return None
        if not key:return None   # base ingest will hold it (coordinates/input); reservation stays as is
        if (key[0]==reservation['content_sha256'] and sorted(map(str,key[1]))==sorted(map(str,reservation['ids']))
                and key[3]==reservation.get('link_sha256')):return None
        return dict(id=batch_id,status='conflict',reservation_kept=True,
            reason=('이 batch_id는 이전 시도가 중단돼 다른 내용·제품 ID·연결로 예약돼 있습니다. 기존 예약은 그대로 두었습니다. '
                    '이전 파일을 같은 batch_id로 다시 보내 마무리하거나, 새 자료는 새 batch_id로 보내세요.'))

    def _commit(self,batch_id,content_sha,note,link=None,id_mode=None):
        """Commit point after the base stored files + ledger: processing state first, then clear the reservation."""
        with self.lock():
            manifest_path=self.state/'batches'/batch_id/'manifest.json'
            manifest=read(manifest_path)
            if id_mode and 'id_mode' not in manifest:
                manifest['id_mode']=id_mode
                write(manifest_path,manifest)
            ledger=read(self.state/'batch_ledger.json',{'batches':[]})
            if batch_id not in ledger['batches']:          # base stored manifest but the ledger write failed
                ledger['batches'].append(batch_id);write(self.state/'batch_ledger.json',ledger)
            if not self._processing_path(batch_id).exists():
                write(self._processing_path(batch_id),dict(batch_id=batch_id,stage='drift_pending',
                    content_sha256=content_sha,link_sha256=link,id_mode=id_mode,committed_at=now(),
                    duplicate_note=note,attempts=[]))
            index=self._index()
            if index['reservations'].pop(batch_id,None) is not None:write(self.state/'dedup_index.json',index)

    def ingest(self,frame,batch_id=None,label_source=None,coordinates_confirmed=False):
        batch_id=clean_id(batch_id or new_id('batch'))
        with self._exclusive('ingest'):
            key=None;key_error='coordinates_confirmed 없음(좌표·스케일 정합 미확인)';id_error=None
            try:validate_product_ids(frame)
            except ProductIdError as exc:id_error=str(exc)
            if coordinates_confirmed and not id_error:
                try:key=self._dedup_key(frame,batch_id,label_source);key_error=None
                except (ValueError,TypeError,KeyError) as exc:key_error=str(exc)   # base ingest records the hold reason
            if id_error:key_error=id_error
            folder=self.state/'batches'/batch_id
            try:
                previous=read(folder/'manifest.json',{}) if folder.exists() else {}
                if not isinstance(previous,dict):raise ValueError('배치 manifest 형식 오류')
                registered=batch_id in read(self.state/'batch_ledger.json',{'batches':[]})['batches']
                if previous.get('status')!='accepted' and (registered or self._processing_path(batch_id).exists()):
                    raise ValueError('수집 이력이 있으나 accepted manifest가 없거나 손상됨')
            except (ValueError,OSError,TypeError,KeyError) as exc:
                return dict(id=batch_id,status='rejected',reason=f'기존 수집 근거를 검증할 수 없습니다({exc}). 재수집하지 않았습니다.')
            if previous.get('status')=='accepted':
                if key is None:
                    # Never answer 'already completed / resumed' for a re-sent file that could not be compared.
                    return dict(id=batch_id,status='rejected',
                        reason=f'이미 수집된 batch_id의 재전송 자료를 검증할 수 없어 비교하지 않았습니다({key_error}). '
                               '기존 배치는 그대로입니다. 미완료 단계 재개는 resume 명령을 쓰세요.')
                return self._retry(batch_id,key,frame)
            # A reservation left by an interrupted attempt belongs to that exact content and those product IDs.
            conflict=self._reservation_conflict(batch_id,key)
            if conflict:return conflict
            if id_error:
                if folder.exists() or self._index()['reservations'].get(batch_id):
                    return dict(id=batch_id,status='rejected',reason=id_error)   # keep the earlier attempt untouched
                return self._held(batch_id,id_error)
            if folder.exists():
                # Nothing was committed (held or interrupted before the manifest): archive and process again.
                archive=self.state/'held_archive'/f"{batch_id}-{datetime_stamp()}"
                archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(folder),str(archive))
            note=None
            if key:
                reason,note=self._duplicate_check(key,self._index(),batch_id)
                if reason:return self._held(batch_id,reason)
                self._reserve(batch_id,key,note)
            self._defer_detect=True
            try:result=super().ingest(frame,batch_id,label_source,coordinates_confirmed)
            finally:self._defer_detect=False
            if result['status']!='accepted':
                if key:self._release(batch_id)
                return result
            result.pop('drift',None)
            self._commit(batch_id,key[0] if key else None,note,key[3] if key else None,
                         ('explicit_product_id' if key[2] else 'no_product_id_compat') if key else None)
            if note:result['duplicate_note']=note
            return self._finish(result)

    def resume(self,batch_id):
        """Re-run unfinished stages of an accepted batch (incl. an interrupted commit). Never re-stores it."""
        batch_id=clean_id(batch_id)
        with self._exclusive('ingest'):
            try:
                m=read(self.state/'batches'/batch_id/'manifest.json')
                if not isinstance(m,dict) or m.get('status')!='accepted':
                    raise ValueError('수집 완료 manifest 근거 없음')
            except (ValueError,OSError) as exc:
                return dict(id=batch_id,status='rejected',reason=f'저장 배치 근거를 검증할 수 없어 재개 거부: {exc}')
            return self._retry(batch_id,None)

    def baseline_status(self):
        """Read-only check: current baseline exists, matches active model versions, files unchanged.
        Never creates a baseline (unlike baseline())."""
        reasons=[];pointer=read(self.state/'baseline_pointer.json')
        if not pointer:return dict(valid=False,reasons=['기준선 포인터 없음'],active=self.active())
        folder=self.state/'baselines'/pointer['id'];m=read(folder/'manifest.json')
        if not m:return dict(valid=False,baseline=pointer['id'],reasons=['기준선 manifest 없음'],active=self.active())
        if m['versions']!=self.active():reasons.append('기준선 모델 버전이 현재 운영 모델과 다름')
        for name,key in [('reference_products.csv','reference_sha256'),('reference_patterns.csv','patterns_sha256')]:
            if not (folder/name).exists() or digest(folder/name)!=m[key]:reasons.append(f'{name} 해시 불일치')
        return dict(valid=not reasons,baseline=pointer['id'],versions=m['versions'],active=self.active(),reasons=reasons)

    def pending_batches(self):
        out=[]
        for p in sorted((self.state/'batches').glob('*/processing.json')):
            r=read(p)
            if r.get('stage')!='completed':out.append(dict(batch_id=r['batch_id'],stage=r['stage'],error=r.get('error')))
        for bid,r in sorted(self._index()['reservations'].items()):
            if self._processing_path(bid).exists():continue
            m=read(self.state/'batches'/bid/'manifest.json') or {}
            if m.get('status')=='accepted':
                out.append(dict(batch_id=bid,stage='commit_incomplete',error=None,
                    message='파일은 저장됐지만 수집 확정이 끝나지 않음. 같은 batch_id로 다시 보내거나 resume 실행'))
            else:
                out.append(dict(batch_id=bid,stage='reserved_not_stored',error=None,
                    message='중복 예약만 있고 파일 저장 전 중단. 같은 batch_id로 원본 파일을 다시 보내세요'))
        return out

    def _stored_link(self,batch_id,recorded_mode=None,frame=None):
        """Link hash recomputed from the stored, hash-verified products.csv (does not trust processing.json).

        A/C is taken from processing or collection manifest (id_mode), never guessed from ID shape.
        Before id_mode existed, every stored ID must have a positive registration for this batch to prove A.
        Absence or partial loss cannot prove C: ambiguous legacy evidence is explicitly rejected.
        Returns (link or None for contract C, explicit). Any missing/unreadable/changed evidence -> ValueError."""
        folder=self.state/'batches'/batch_id
        try:
            m=read(folder/'manifest.json')
            if not m or 'products.csv' not in m.get('files',{}):raise ValueError('저장 제품 파일 기록 없음')
            if not (folder/'products.csv').is_file():raise ValueError('저장 제품 파일 없음')
            if digest(folder/'products.csv')!=m['files']['products.csv']:raise ValueError('저장 제품 파일 해시 불일치')
            p=pd.read_csv(folder/'products.csv',dtype={'record_id':str},keep_default_na=False,float_precision='round_trip')
            if not {'record_id','fingerprint'}<=set(p):raise ValueError('저장 제품 파일 열 누락')
        except ValueError:raise
        except Exception as exc:raise ValueError(f'저장 제품 파일 읽기 실패: {type(exc).__name__}: {exc}') from None
        mode=recorded_mode or m.get('id_mode')
        if recorded_mode and m.get('id_mode') and recorded_mode!=m['id_mode']:
            raise ValueError('처리 상태와 수집 manifest의 ID 계약이 다름')
        if mode is None:
            try:
                registered=self._index()['ids']
                if not isinstance(registered,dict):raise ValueError('ID 색인 형식 오류')
            except Exception as exc:
                raise ValueError(f'이전 배치의 ID 계약 색인을 검증할 수 없음: {type(exc).__name__}') from None
            if not all(registered.get(i)==batch_id for i in p.record_id.tolist()):
                raise ValueError('이전 배치의 ID 계약 근거 부족: 미등록 ID를 C 계약으로 추정하지 않음. 수집 출처 확인 필요')
            mode='explicit_product_id'
        if mode not in ('explicit_product_id','no_product_id_compat'):
            raise ValueError('알 수 없는 수집 ID 계약')
        if mode=='no_product_id_compat':return None,False
        label=p['label'].astype(int).astype(str) if 'label' in p else pd.Series(['NA']*len(p))
        rows=pd.DataFrame(dict(id=p.record_id.astype(str),fp=p.fingerprint,label=label.to_numpy()))
        rows=rows.sort_values(['id','fp','label'],kind='stable').reset_index(drop=True)
        return hashlib.sha256(rows.to_csv(index=False,lineterminator='\n').encode('utf-8')).hexdigest(),True

    def _verify_resume_files(self,batch_id,manifest,proc):
        """Verify immutable collection files and the recorded ID contract before resume mutates state."""
        try:
            if manifest.get('id')!=batch_id or manifest.get('dataset')!=self.dataset:
                raise ValueError('배치 ID/데이터셋 기록 불일치')
            files=manifest.get('files')
            if not isinstance(files,dict) or not {'products.csv','patterns.csv','predictions.csv'}<=set(files):
                raise ValueError('수집 파일 해시 기록 누락')
            folder=self.state/'batches'/batch_id
            for name,expected in files.items():
                if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):
                    raise ValueError('수집 파일 이름 형식 오류')
                path=folder/name
                if not path.is_file() or digest(path)!=expected:raise ValueError(f'{name} 누락/해시 불일치')
            # A stored-but-not-committed batch may only have its original reservation contract.
            evidence=proc if proc is not None else self._index()['reservations'].get(batch_id,{})
            stored,_=self._stored_link(batch_id,evidence.get('id_mode'))
            if 'link_sha256' in evidence and evidence['link_sha256']!=stored:
                raise ValueError('처리 기록의 연결 해시가 저장 제품 파일과 다름')
        except Exception as exc:
            raise ValueError(f'저장 배치 자료 검증 실패: {type(exc).__name__}: {exc}') from None

    def _retry(self,batch_id,key,frame=None):
        try:
            proc=read(self._processing_path(batch_id))
            if proc is not None and (not isinstance(proc,dict) or 'stage' not in proc):
                raise ValueError('처리 상태 형식 오류')
        except (ValueError,OSError) as exc:
            return dict(id=batch_id,status='rejected',reason=f'처리 상태 근거 손상: {exc}. 기존 배치를 유지합니다.')
        manifest=read(self.state/'batches'/batch_id/'manifest.json')
        if key is None:
            try:self._verify_resume_files(batch_id,manifest,proc)
            except ValueError as exc:return dict(id=batch_id,status='rejected',reason=str(exc))
        if proc is None:
            reservation=self._index()['reservations'].get(batch_id)
            if reservation is None:
                return dict(id=batch_id,status='rejected',reason='처리 상태 및 예약 근거가 없어 완료/재개를 확인할 수 없습니다. 기존 배치를 유지합니다.')
            conflict=self._reservation_conflict(batch_id,key)
            if conflict:return conflict
            # Interrupted commit: finish it (ledger/processing state), then run the later stages.
            self._commit(batch_id,reservation['content_sha256'],reservation.get('note'),
                         reservation.get('link_sha256'),reservation.get('id_mode'))
            proc=read(self._processing_path(batch_id))
        if key:
            if proc.get('content_sha256') and key[0]!=proc['content_sha256']:return self._conflict(batch_id,self.LINK_CONFLICT)
            # Compare with the link recomputed from the stored product file, for every batch including ones
            # committed before link_sha256 existed. No usable evidence -> refuse instead of answering success.
            try:stored,_=self._stored_link(batch_id,proc.get('id_mode'),frame)
            except ValueError as exc:
                return dict(id=batch_id,status='rejected',
                    reason=f'최초 수집 제품 기록을 검증할 수 없어 재전송을 비교하지 못했습니다({exc}). 기존 배치는 그대로입니다.')
            if key[3]!=stored:return self._conflict(batch_id,self.LINK_CONFLICT)
            if 'link_sha256' in proc and proc['link_sha256']!=stored:
                return dict(id=batch_id,status='rejected',reason='처리 기록의 연결 해시가 저장 제품 파일과 다릅니다. 기존 배치는 그대로입니다.')
        if proc['stage']=='completed':
            return dict(manifest,processing=dict(stage='completed',retry='already_completed',
                message='이미 수집·감지·안내가 끝난 배치입니다. 다시 처리하지 않았습니다.',summary=proc.get('summary')),
                drift=proc.get('drift'))
        result=dict(manifest)
        if proc.get('duplicate_note'):result['duplicate_note']=proc['duplicate_note']
        return self._finish(result,resumed=True)

    def _drift_containing(self,batch_id):
        found=[read(p) for p in (self.state/'drift').glob('*.json')]
        found=[d for d in found if d and batch_id in d.get('batch_ids',[])]
        return sorted(found,key=lambda d:d['created_at'])[-1] if found else None

    def _finish(self,result,resumed=False):
        bid=result['id'];path=self._processing_path(bid);proc=read(path)
        proc['attempts'].append(dict(at=now(),resumed=resumed))
        if resumed:
            # A drift record may exist whose consumption/history was never written (stop between the two
            # writes). Apply it once by drift id before reusing it, so the next window does not re-consume.
            self.reconcile_monitor()
        if proc.get('drift') is None:
            try:
                # A later batch's drift run may already have consumed this batch.
                drift=self._drift_containing(bid) if resumed else None
                drift=drift or self._run_detect()
            except Exception as exc:
                proc.update(stage='drift_failed',error=f'{type(exc).__name__}: {exc}');write(path,proc)
                result.update(drift=None,processing=dict(stage='drift_failed',error=proc['error'],
                    message='수집은 완료(장부 등록)됐고 분포 감지 단계가 실패했습니다. 원인을 해결한 뒤 같은 batch_id로 '
                            '다시 보내거나 resume 명령을 실행하세요. 다른 batch_id로 다시 보내지 마세요.'))
                return result
            proc.update(drift=drift,stage='advice_pending',error=None);write(path,proc)
        result['drift']=proc['drift']
        try:result=self._advise(result)
        except Exception as exc:
            # The orchestration hook may persist advice_context; preserve it and this attempt.
            attempts=proc['attempts'];proc.update(read(path));proc['attempts']=attempts
            proc.update(stage='advice_failed',error=f'{type(exc).__name__}: {exc}');write(path,proc)
            result['processing']=dict(stage='advice_failed',error=proc['error'],
                message='수집·분포 감지는 완료됐고 검사 안내 생성이 실패했습니다. 같은 batch_id로 다시 보내거나 resume 명령을 실행하세요.')
            return result
        attempts=proc['attempts'];proc.update(read(path));proc['attempts']=attempts
        summary=dict(drift_status=(proc['drift'] or {}).get('status'),
                     inspection_priority=(result.get('inspection_priority') or {}).get('status'),
                     reinspection_count=(result.get('inspection_advice') or {}).get('reinspection_count'))
        proc.update(stage='completed',completed_at=now(),error=None,summary=summary);write(path,proc)
        try:   # a reservation left by an interrupted commit is no longer needed once processing is complete
            with self.lock():
                index=self._index()
                if index['reservations'].pop(bid,None) is not None:write(self.state/'dedup_index.json',index)
        except OSError:pass
        result['processing']=dict(stage='completed',resumed=resumed)
        return result

    def _advise(self,result):
        if self.dataset=='cn7':return self._cn7_priority(result)
        return self._rg3_advice(result)

    # ---- F06: drift calibration bound to its policy ----
    def detect(self):
        if self._defer_detect:return {'status':'deferred'}
        return self._run_detect()

    def _calibration_signature(self,baseline_id):
        import pipeline_runtime
        return dict(baseline_id=baseline_id,quantile=float(self.policy['drift_quantile']),
                    repeats=int(self.policy['bootstrap_repeats']),runtime_code_sha256=digest(pipeline_runtime.__file__))

    def _run_detect(self):
        pointer=read(self.state/'baseline_pointer.json')
        if pointer:
            folder=self.state/'baselines'/pointer['id'];signature=self._calibration_signature(pointer['id'])
            with self.lock():
                for f in folder.glob('calibration_*.json'):
                    if (read(f) or {}).get('signature')!=signature:
                        archive=folder/'calibration_archive';archive.mkdir(exist_ok=True)
                        shutil.move(str(f),str(archive/f'{f.stem}-{datetime_stamp()}.json'))
        result=BaseOperations.detect(self)
        calibration=result.get('calibration')
        if calibration:
            signature=self._calibration_signature(result['baseline'])
            cache=self.state/'baselines'/result['baseline']/f"calibration_{calibration['n_eff']}_{calibration['repeats']}.json"
            with self.lock():
                if cache.exists():write(cache,dict(read(cache),signature=signature))
                record=self.state/'drift'/f"{result['id']}.json"
                if record.exists():write(record,dict(read(record),calibration_signature=signature))
            result['calibration_signature']=signature
        return result

    # ---- RG3 advice ----
    def _rg3_advice(self,result):
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
            if u=='pending':
                r['message']='분포 감지 표본을 모으는 중이라 배치 변화 판단이 아직 없습니다(판단 대기). '+r['message']
        zones=drift_zones(result.get('drift'))
        levels=pd.Series([r['evidence_level'] for r in rows]).value_counts().to_dict()
        uncertainty=pd.Series([r['uncertainty_level'] for r in rows]).value_counts().to_dict()
        advice=dict(dataset='rg3',baseline_id=manifest['id'],created_at=now(),
            drift=result['drift'],batch_advice=batch_level,level_counts=levels,
            uncertainty_counts=uncertainty,uncertainty_zones=zones,
            uncertainty_meaning=('높음·중간·낮음은 판단 불확실 정도이며 불량 위험도·불량 확률이 아님. '
                                 '판단 대기는 분포 감지 표본 부족으로 아직 판단하지 않은 상태이며 낮음(정상)이 아님'),
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

    # ---- CN7 inspection budget (F05) ----
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
        rule=self.policy['cn7_tie_rule']
        flag,rank,k,cutoff,ties=budget_flags(scores,self.policy['inspection_fraction'],products.record_id.astype(str),rule,
                                             products.fingerprint.astype(str))
        table=pd.DataFrame(dict(record_id=products.record_id.astype(str),fingerprint=products.fingerprint,
            risk_score=scores,rank=rank,inspect=flag.astype(int),model_version=version))
        table=table.sort_values(['rank','record_id'],kind='stable')
        table.to_csv(folder/'inspection_priority.csv',index=False,encoding='utf-8-sig')
        tie_note=('정확히 예산 수만 표시. 경계 동점은 입력 패턴 지문(fingerprint)→record_id 순으로 선택(평가도 같은 규칙)'
                  if rule=='exact_k' else '경계 동점은 모두 표시하므로 실제 검사 수가 예산을 넘을 수 있음(over_budget에 기록). '
                  'all_ties 결과는 열람용이며 모델 선정·승격 비교에 쓰지 않음')
        info=dict(status='ranked',model_kind=kind,model_version=version,unit='product',batch_products=len(products),
            batch_patterns=int(products.fingerprint.nunique()),
            inspection_fraction=self.policy['inspection_fraction'],budget_k=k,flagged=int(flag.sum()),inspected=int(flag.sum()),
            score_cutoff=cutoff,**ties,
            meaning='배치 안에서 위험 점수 상위 제품을 검사 대상으로 표시. 점수는 위험 이력 패턴 순위이며 불량 확률이 아님',
            note=tie_note+'. 배치 예산은 제품 수 기준, 평가 예산은 고유 패턴 수 기준이라 반복 제품이 많으면 같은 비율의 뜻이 다름. '
                 '예산 비율은 현장 검사 여력에 맞춰 사전 확정')
        write(folder/'inspection_priority.json',info)
        (folder/'inspection_priority.txt').write_text('\n'.join([info['meaning'],info['note'],
            f"검사 대상 {info['flagged']}건 / 배치 {len(products)}건 (예산 {k}건, 동점 규칙 {rule}, 모델 {version})"]
            +[f"{r.record_id}: 순위 {r.rank}, 점수 {r.risk_score:.4f}" for r in table[table.inspect.eq(1)].itertuples()]),encoding='utf-8')
        write(folder/'decision_manifest.json',dict(files={p.name:digest(p) for p in folder.glob('inspection_priority.*')}))
        result['inspection_priority']=dict(info,csv=str(folder/'inspection_priority.csv'))
        return result

    def _budget_metrics(self,evaluation_id,candidate_bundle,current_bundle):
        p=pd.read_csv(self.state/'evaluations'/clean_id(evaluation_id)/'patterns.csv',float_precision='round_trip')
        y=p.label.to_numpy();x=p[feature_columns()];f=self.policy['inspection_fraction'];rule=self.policy['cn7_tie_rule']
        fp=p.fingerprint.astype(str).to_numpy()
        cm=budget_metrics(y,score(candidate_bundle,x),f,rule,fp)
        om=budget_metrics(y,score(current_bundle,x),f,rule,fp) if current_bundle is not None else None
        return cm,om

    # ---- F07: evaluation purpose record integrity ----
    def register_evaluation(self,frame,label_source,purpose='independent',evaluation_id=None):
        eid=super().register_evaluation(frame,label_source,purpose,evaluation_id)
        folder=self.state/'evaluations'/eid
        with self.lock():
            write(folder/'manifest_integrity.json',dict(manifest_sha256=digest(folder/'manifest.json'),
                patterns_sha256=read(folder/'manifest.json')['sha256'],created_at=now(),
                note='평가 파일과 용도(purpose)·패턴 목록 기록을 함께 고정'))
        return eid

    def _verify_evaluation_manifest(self,evaluation_id):
        folder=self.state/'evaluations'/clean_id(evaluation_id)
        integrity=read(folder/'manifest_integrity.json')
        if not integrity:
            raise ValueError('평가 용도 기록(manifest) 무결성 정보 없음: 이 기능 도입 전에 등록한 평가 자료는 다시 등록하세요')
        if not (folder/'manifest.json').exists() or digest(folder/'manifest.json')!=integrity['manifest_sha256']:
            raise ValueError('평가 용도 기록(manifest)이 등록 후 변경됨: 평가·승격 근거로 사용할 수 없음')
        return integrity['manifest_sha256']

    def evaluate(self,candidate,evaluation_id):
        manifest_sha=self._verify_evaluation_manifest(evaluation_id)
        result=super().evaluate(candidate,evaluation_id)
        if self.dataset=='cn7' and result['kind'] in ('lr','rf','ocsvm') and self.policy['cn7_decision_mode']=='budget':
            b,_=self.load_model(candidate);old=self.load_model(result['current'])[0] if result['current'] else None
            cm,om=self._budget_metrics(evaluation_id,b,old)
            result['candidate_budget_metrics']=cm;result['current_budget_metrics']=om
            # Budget mode replaces the fixed-threshold comparison by same-budget risk capture.
            result['reasons']=[r for r in result['reasons'] if r!='F1 개선·재현율 유지·오탐률 상한 조건 미충족']
            enough='평가 정상/위험 고유 패턴 부족' not in result['reasons']
            if self.policy['cn7_tie_rule']!='exact_k':
                result['reasons'].append('all_ties 평가는 참고 전용: 실제 검사 수가 예산과 달라 선정·승격 근거로 쓸 수 없음(exact_k 필요)')
            reachable=min(cm['inspected'],cm['positives'])   # most risks findable with the inspections actually made
            cm['capture_of_reachable']=cm['found_TP']/reachable if reachable else None
            if enough and cm['found_TP']<self.policy['min_budget_capture']*reachable:
                result['reasons'].append(f"검사 예산 내 위험 발견 {cm['found_TP']}/{reachable}(예산 내 최대) — 최소 비율 {self.policy['min_budget_capture']} 미달")
            if enough and om is not None and cm['found_TP']<=om['found_TP']:
                result['reasons'].append('동일 검사량 위험 발견의 개선 없음')
            result['passed']=not result['reasons'];result['decision_mode']='budget'
            result['acceptance_policy_note']=self.policy['acceptance_policy_note']
        elif self.dataset=='cn7' and result['kind'] in ('lr','rf','ocsvm'):
            result['reasons'].extend(usability(result['candidate_metrics'],self.policy))
            result['passed']=not result['reasons'];result['decision_mode']='threshold'
            result['acceptance_policy_note']=self.policy['acceptance_policy_note']
        result['evaluation_manifest_sha256']=manifest_sha
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
                if report['candidate_budget_metrics'].get('tie_rule')!='exact_k':
                    raise ValueError('exact_k가 아닌 검사 예산 평가로는 승격할 수 없음(all_ties는 참고 전용)')
            else:
                reasons=usability(report['candidate_metrics'],self.policy)
                if reasons:raise ValueError('CN7 사용 기준 미충족: '+str(reasons))
        if report and report['evaluation_id'] in self.selection_evaluations():
            raise ValueError('모델 선정에 사용한 평가 자료로는 승격할 수 없음: 별도 평가 자료 등록 필요')
        if report:
            current=self._verify_evaluation_manifest(report['evaluation_id'])
            if report.get('evaluation_manifest_sha256')!=current:
                raise ValueError('평가 당시와 평가 용도 기록이 다름(또는 기록 없음): 다시 평가하세요')
        return super().promote(assessment_id)

    def select_cn7(self,candidates,evaluation_id):
        """Same independent evaluation, qualified candidates only, no activation."""
        if self.dataset!='cn7':raise ValueError('CN7 전용 모델 선정')
        if self.policy['cn7_decision_mode']=='budget' and self.policy['cn7_tie_rule']!='exact_k':
            raise ValueError('모델 선정은 exact_k 동점 규칙에서만 가능(all_ties는 실제 검사 수가 달라 참고 전용)')
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
                report['inspection_budget_reference']=budget_metrics(evaluation.label.to_numpy(),
                    score(bundle,evaluation[feature_columns()]),self.policy['inspection_fraction'],self.policy['cn7_tie_rule'],
                    evaluation.fingerprint.astype(str).to_numpy())
        eligible=[r for r in reports if r['passed']]
        if self.policy['cn7_decision_mode']=='budget':
            # Same budget: more risks found first; ties prefer the simpler/explainable family.
            order={'lr':0,'rf':1,'ocsvm':2}
            eligible.sort(key=lambda r:(-(r.get('candidate_budget_metrics') or r.get('inspection_budget_reference') or {}).get('found_TP',0),
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


def datetime_stamp():
    return time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'-'+os.urandom(3).hex()
