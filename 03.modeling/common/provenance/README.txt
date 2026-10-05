pipeline_runtime.py 출처 보관본 (2026-10-05)

왜 있나
- 저장된 LR 실행(output/logistic/<cn7|rg3>/<run>/run_config.json)은 실행 당시 pipeline_runtime.py 해시를 기록한다.
- 2026-10-05에 감지 소비 장부 복구(R01) 때문에 pipeline_runtime.py의 Operations 클래스만 수정했다.
- 과거 실행 기록은 고치지 않는다. 대신 그 해시에 해당하는 원본을 여기 보관한다.

감사 방법 (03.modeling/tests/audit_logistic_runs.py)
1) 기록된 해시 == 현재 파일 해시 → current
2) 아니면 pipeline_runtime_<해시 앞 12자리>.py 보관본의 전체 해시가 기록과 같은지 확인(출처 보존)
3) 보관본과 현재 파일의 학습 경로 정의(Operations 클래스를 제외한 모든 문장)가 AST 기준 동일한지 확인
   → 같으면 LR 학습·평가 결과는 현재 코드에서도 같은 정의로 재현된다는 근거
   → 다르면 감사 실패: LR 실행을 새로 해야 함

보관본은 수정·삭제하지 않는다. 이 해시 확인은 기록 불일치 탐지 수단이며 악의적 변조를 막는 인증이 아니다.

보관본 목록
pipeline_runtime_bfedfd6d0c66.py  2026-10-03 LR 실행 당시 원본(감지 복구 수정 전)

노트북 학습 정의 (2026-10-05 추가, Codex 검수 지적 반영)
- LR 학습은 ocsvm.ipynb의 pipeline_library 셀(fit_features·transform_features·rank_candidates·load_data)에도 의존한다.
- 2026-10-03 LR 실행은 이 셀들의 해시를 기록하지 않았다. 따라서 해시만으로 실행 당시와 같다고 증명할 수 없다.
- 감사는 두 가지를 구분해 확인한다.
  1) 저장 모델의 Test 예측 재현(기존)
  2) 선택된 설정을 현재 코드로 개발 패턴에 다시 학습 → 저장 모델과 개발+Test 예측 차이 1e-8 이하, Test 판정 완전 일치
     (pipeline_runtime + 노트북 학습 정의 전체 경로를 실제로 실행해 확인. 단, 탐색한 240개 설정 전부는 아님)
- notebook_library_snapshot.json: 2026-10-05 기준 셀 해시(실행 당시 기록 아님). 이후 변경 여부만 보고한다.
