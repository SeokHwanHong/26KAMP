# 팀장 권장안 1~5 확정 운영 정책

2026-10-06 사용자 승인으로 1~5 권장안을 확정했다. 공식 운영 클래스는 `governance_runtime.Operations`다.
`pipeline → decision → workflow → governance` 계층을 사용한다. 하위 클래스 직접 호출은 연구·기존 기능 검사 경로다.
CLI, 운영 노트북, 검사 실행기는 governance 계층을 사용한다.

## 1. CN7: RF 초기 기준과 독립 제품 검사 후 명시적 전환

초기 검사 역할은 RF, budget/exact_k다. 모델 선정 보고서만으로 운영 역할을 변경하지 않는다.
실제 제품 ID·입력·검사 라벨이 독립 패턴 평가와 일치하는지 확인한 뒤, 같은 제품 검사 수에서 후보와 현재 검사 모델의 발견 수를 비교한다.
제품 검사 성과 통과와 담당자의 승격 승인이 모두 있어야 교체한다.

`registry.inspection_role`에 종류·버전·검사 비율·동점 규칙·승인·제품 평가·이전 역할을 저장한다.
명시적 승격에서 모델과 검사 역할이 같은 전환 저널로 이동한다. RF→LR 등의 역할 변경도 실제 순위 모델에 반영한다.
롤백은 모델과 이전 검사 역할을 함께 되돌린다. 실패 전환은 기존 resume-transition으로 복구한다.
IF 보조 모델의 승격은 검사 역할을 바꾸지 않는다.

제품 평가는 실제 제공 라벨 기준의 기능이다. 라벨 진위 자체를 자동 인증하지 않으며, 과거 Test/선정 자료를 독립 평가로 바꾸지 않는다.

## 2. RG3: 우선검사와 무작위 표본검사 병행

불확실성 안내와 실제 검사 계획을 구분한다. review의 전체 높은 불확실성을 전수검사 지시로 해석하지 않는다.

- 우선검사: 범위 밖·정보 부족을 우선하고 동일 조건은 제품 ID로 안정 정렬.
- 무작위 검사: 우선검사 외 제품에서 비복원 추출.
- 두 목록은 겹치지 않으며 전체 검사 수는 `ceil(제품 수 × fraction)`이다.
- 실제 검사 예산이 2개 미만이면 두 방식을 병행할 수 없어 `held_budget`으로 보류한다.
- 원래 제품 정답을 검사 대상 선정에 사용하지 않는다.
- 배치 ID·정책 버전 기반 seed를 저장한다. 같은 근거에서 계획을 재현할 수 있다.

단계별 `waiting/normal/watch/review` 설정이 필요하다.
`fraction`은 전체 검사 비율(0 초과 1 이하), `random_share`는 선정 검사 수 중 무작위 비중(0 초과 1 미만)이다.
비율이 없으면 `held_configuration`으로 기록하고 실제 검사 대상을 임의로 만들지 않는다.

산출물: `inspection_plan.json`, `inspection_advice.{json,csv,txt}`의 실제 선정 표시.
기존 reinspection_recommended는 근거상 권고, inspection_planned/inspection_method는 실제 계획이다.
개별 불량 확률이나 공정 조건 추천을 새로 제공하지 않는다.

## 3. 확인된 정상 공정만 별도 승인 참조로 편입

`update-reference`는 원천 제품 ID·시간대 포함 생산 시각·공정 버전·메타데이터 출처·실제 정상 검사 라벨을 요구한다.
같은 공정 버전의 자료를 사용하며 고유 정상 표본 조건을 충족해야 한다.
평가 자료를 참조에 편입하거나 과거 위험 패턴을 새 정상으로 흡수하지 않는다.

기존 입력 기준을 보존하고 새 참조를 별도 버전으로 만든다. 이전 기준으로 계산한 병행 변화 비교를 저장한 뒤 승인 기록과 함께 전환한다.
모델 CT/승격은 승인된 입력 참조를 유지하고 모델 출력 기준만 다시 계산한다.
`baseline` 명령으로 승인 참조를 초기 개발 참조로 덮어쓰지 않는다.
`reference-status`에서 원천·이전 참조·자료 크기를 조회하고 `rollback-reference`로 명시적으로 복구한다.

새 입력 참조 제품 분포는 확인된 정상 제품이다. 구간 안내 패턴 표에는 확인 정상과 영구 위험 이력을 함께 보존하므로 그 비율을 제품 불량률로 해석하지 않는다.
참조 갱신의 실제 현장 효과는 독립 공정 검사로 검증해야 한다.

## 4. 영구 위험 이력과 학습 대상 분리

기본 학습 범위는 cumulative를 유지한다. 원본 제품·패턴·모델 corpus와 별도 `governance/risk_history/<ID>`를 보존한다.
최근 적응이 필요하면 정책에 `mode=recent`, 시간대 포함 start/end, process_version, max_patterns를 지정한다.

생산 시각은 입력 CSV의 실제 열을 제품 ID와 연결해 production_context.json에 저장한다.
행 번호나 수집 시각으로 기간을 추정하지 않는다. 시간대가 없거나 원천 ID가 없으면 시간 연결을 거부한다.
메타데이터가 없는 대회 자료는 recent 학습에 사용할 수 없다.

학습 대상은 지정 기간·공정 버전의 자료로 제한해도, 선택된 패턴의 과거 위험 이력은 최대 라벨로 유지한다.
평가 중복, 정상/위험 표본 부족, 학습 상한 초과는 학습 전 보류한다.
추정기 파라미터와 부모 임계값을 유지하고 선택 학습 자료에서 변환을 다시 적합한다.

## 5. 승인 담당자와 근거 기록

정책에 ct/promote/reference/rollback 담당자 목록을 등록한다.
등록되지 않은 담당자는 승인할 수 없고, 승인 ID가 없으면 CT·승격·롤백·참조 갱신을 실행하지 않는다.

CT와 참조 승인 근거:

- normal_process_confirmed: 실제 정상 공정 검사 확인(true).
- input_error_excluded: 입력/센서/좌표 오류를 확인·배제(true).
- changed_representations: 바뀐 변수·표현의 목록.
- label_source: 실제 검사 라벨 출처.
- evaluation_plan: 분리 평가 계획.

승격·롤백은 승인 사유를 기록한다. 모델·기준선·정책·배치 자료·생산 메타데이터·평가 근거의 해시와 연결한다.
승인 후 대상이나 근거가 바뀌면 재승인을 요구한다.
이는 로컬 담당자/근거 기록과 실행 조건 검사다. 외부 계정 인증·조직 권한 시스템 연동은 포함하지 않는다.

## 명령과 준비 파일

실제 운영에 적용하기 전에 `GOVERNANCE_SETTINGS.template.json`을 별도 파일로 복사해 값과 담당자를 기입한다.
템플릿은 안전하게 approvers={}, rg3_sampling=null, cumulative로 두었다. 테스트용 담당자와 비율을 실제 설정에 넣지 않았다.

```powershell
python 03.modeling/pipeline_cli.py --dataset rg3 --state-root tmp/rehearsal governance-status
python 03.modeling/pipeline_cli.py --dataset rg3 --state-root tmp/rehearsal configure-governance --settings-json settings.json --actor 담당자 --reason "팀 정책 승인"
python 03.modeling/pipeline_cli.py --dataset rg3 --state-root tmp/rehearsal ingest --input batch.csv --batch-id B1 --label-source "검사 출처" --coordinates-confirmed --production-time-column production_time --process-version P1 --metadata-source "생산 시스템"
python 03.modeling/pipeline_cli.py --dataset rg3 --state-root tmp/rehearsal approve-action --action-type ct --payload-json payload.json --evidence-json evidence.json --actor 승인담당자
```

CT payload의 필드: kind, batch_ids, drift_id, cause, base_version(null 또는 현재 버전).
승격 payload: assessment_id, product_validation_id(CN7은 필수, RG3는 null 가능).
참조 갱신 payload: batch_ids, reason.
모델 롤백 payload: kind, reason.
참조 롤백 payload: batch_ids=[], reason, rollback_baseline(현재 참조 원천의 previous).
실행 명령의 인자와 승인 payload가 정확히 일치해야 한다.

```powershell
python 03.modeling/pipeline_cli.py --dataset rg3 --state-root tmp/rehearsal retrain --kind lr --batch-ids B1 B2 --drift-id DRIFT_ID --cause "확인된 원인" --approval-id APPROVAL_ID
python 03.modeling/pipeline_cli.py --dataset cn7 --state-root tmp/rehearsal product-validation --assessment-id ASSESSMENT_ID --input independent_products.csv --label-source "독립 검사 출처"
python 03.modeling/pipeline_cli.py --dataset cn7 --state-root tmp/rehearsal promote --assessment-id ASSESSMENT_ID --product-validation-id PRODUCT_VALIDATION_ID --approval-id APPROVAL_ID
python 03.modeling/pipeline_cli.py --dataset rg3 --state-root tmp/rehearsal update-reference --batch-ids NORMAL_BATCH --reason "정상 공정 확인" --approval-id APPROVAL_ID
python 03.modeling/pipeline_cli.py --dataset rg3 --state-root tmp/rehearsal reference-status
python 03.modeling/pipeline_cli.py --dataset rg3 --state-root tmp/rehearsal rollback-reference --reason "참조 복구" --approval-id APPROVAL_ID
```

실제 runtime에 정책 설치·모델 교체·참조 갱신은 이번 코드 검증에서 수행하지 않는다.
공식 경로는 승인 계층이며, 과거 workflow/decision/pipeline 하위 API 테스트는 기반 기능 검증으로 구분한다.
