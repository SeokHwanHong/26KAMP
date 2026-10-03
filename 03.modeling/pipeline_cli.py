"""Explicit local operations entrypoint. Run --help for subcommands."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'common'))
from pipeline_runtime import data,fit_oneclass,fit_supervised
from decision_runtime import Operations
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['cn7','rg3'],required=True)
    p.add_argument('--state-root',type=Path,help='Isolated runtime root for rehearsal')
    p.add_argument('--min-recall',type=float,default=.5)
    p.add_argument('--min-precision',type=float,default=.2)
    p.add_argument('--max-fpr',type=float,default=.2)
    p.add_argument('--cn7-mode',choices=['budget','threshold'],default='budget',help='CN7 판정: 검사 예산(기본) 또는 고정 임계값')
    p.add_argument('--inspection-fraction',type=float,default=.1,help='배치당 검사 비율(검사 예산)')
    p.add_argument('--min-budget-capture',type=float,default=.5,help='평가 시 예산 내 최소 위험 발견 비율')
    p.add_argument('--rg3-history-triggers-recheck',action='store_true',help='RG3 구간 위험 이력만으로도 재검사 권고(기본: 참고 표시만)')
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('status');sub.add_parser('baseline');sub.add_parser('drift')
    initial=sub.add_parser('initial');initial.add_argument('--kind',choices=['if','lr','rf'],required=True)
    activate=sub.add_parser('initialize');activate.add_argument('--version',required=True);activate.add_argument('--reason',required=True)
    batch=sub.add_parser('ingest');batch.add_argument('--input',type=Path,required=True)
    batch.add_argument('--batch-id');batch.add_argument('--label-source');batch.add_argument('--coordinates-confirmed',action='store_true')
    ct=sub.add_parser('retrain');ct.add_argument('--kind',choices=['if','ocsvm','lr','rf'],required=True)
    ct.add_argument('--batch-ids',nargs='+',required=True);ct.add_argument('--drift-id',required=True);ct.add_argument('--cause',required=True)
    ev=sub.add_parser('register-evaluation');ev.add_argument('--input',type=Path,required=True)
    ev.add_argument('--label-source',required=True);ev.add_argument('--purpose',choices=['independent','historical_followup'],default='independent')
    compare=sub.add_parser('evaluate');compare.add_argument('--candidate',required=True);compare.add_argument('--evaluation-id',required=True)
    select=sub.add_parser('select-cn7');select.add_argument('--candidates',nargs=3,required=True);select.add_argument('--evaluation-id',required=True)
    deploy=sub.add_parser('promote');deploy.add_argument('--assessment-id',required=True)
    undo=sub.add_parser('rollback');undo.add_argument('--kind',choices=['if','ocsvm','lr','rf'],required=True);undo.add_argument('--reason',required=True)
    a=p.parse_args()
    if not all(0<=v<=1 for v in [a.min_recall,a.min_precision,a.max_fpr,a.inspection_fraction,a.min_budget_capture]):p.error('성능 기준은 0~1이어야 합니다')
    ops=Operations(a.dataset,a.state_root,dict(min_recall=a.min_recall,min_precision=a.min_precision,max_FPR=a.max_fpr,rg3_history_triggers_recheck=a.rg3_history_triggers_recheck,
        cn7_decision_mode=a.cn7_mode,inspection_fraction=a.inspection_fraction,min_budget_capture=a.min_budget_capture))
    if a.action=='status':result=ops.registry()
    elif a.action=='baseline':result=ops.create_baseline()
    elif a.action=='drift':result=ops.detect()
    elif a.action=='initialize':result=ops.initialize(a.version,a.reason)
    elif a.action=='initial':
        x,y,_,dev,_,_=data(a.dataset)
        b=fit_oneclass(a.dataset,x.loc[dev][y.loc[dev].eq(0)],'if') if a.kind=='if' else fit_supervised(a.dataset,x.loc[dev],y.loc[dev],kind=a.kind)
        result=ops.save_candidate(b,x.loc[dev],y.loc[dev],source={'kind':'fixed baseline; not grid-selected'})
    elif a.action=='ingest':result=ops.ingest(pd.read_csv(a.input,float_precision='round_trip'),a.batch_id,a.label_source,a.coordinates_confirmed)
    elif a.action=='retrain':result=ops.retrain(a.kind,a.batch_ids,a.drift_id,a.cause)
    elif a.action=='register-evaluation':result=ops.register_evaluation(pd.read_csv(a.input,float_precision='round_trip'),a.label_source,a.purpose)
    elif a.action=='evaluate':result=ops.evaluate(a.candidate,a.evaluation_id)
    elif a.action=='select-cn7':result=ops.select_cn7(a.candidates,a.evaluation_id)
    elif a.action=='promote':result=ops.promote(a.assessment_id)
    else:result=ops.rollback(a.kind,a.reason)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
