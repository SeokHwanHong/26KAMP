# 결과보고서 v4 시각화

14개 노트북의 마지막 `결과보고서 v4 시각화` 셀에 40개 그림의 실행 출력을 추가하였다.
원래 셀과 모델 학습·전처리·운영 로직은 보존하였다. 보고서에는 40개 그림 전수를 포함하였다.

## 실행

저장 운영 모델은 scikit-learn 1.9.0 환경에서 읽어야 한다. 이번 실행에 사용한 환경은
`C:/ProgramData/Anaconda3/python.exe`이다. 기존 연구 환경과 버전이 다르므로 전체 학습을
자동으로 다시 실행하지 않는다.

```powershell
& C:/ProgramData/Anaconda3/python.exe reporting/visualizations.py
& C:/ProgramData/Anaconda3/python.exe reporting/execute_notebook_sections.py
& C:/ProgramData/Anaconda3/python.exe reporting/validate_results.py
```

첫 명령은 기존 고정 모델 추론과 저장 실험 CSV에서 그림을 생성한다. RG3 실제 정책 함수는
`output/report_visualizations/20261006_v4/isolated_rg3_policy`에서 격리 호출한다.
승인 설정, 실제 runtime 포인터, 기준선 및 모델은 변경하지 않는다.
두 번째 명령은 추가한 시각화 셀만 실제 커널에서 실행하여 노트북에 출력을 저장한다.
기존 `pipeline_library` 태그를 새 셀에 붙이지 않으므로 라이브러리 로드 시 그림을 실행하지 않는다.

DOCX 작성은 Codex 번들 문서 환경의 Python으로 `reporting/build_report.py`를 실행한다.
PDF는 설치된 Word에서 같은 DOCX를 읽기 전용으로 열어 내보냈다.
공식 6개 장과 휴먼명조 14pt, 본문 160% 줄간격을 적용하였다.

## 결과와 해석

- `output/report_visualizations/20261006_v4/manifest.json`: 전체 그림, 연결 노트북, 해석 및 모델 버전.
- `*_frozen_test.csv`, `frozen_model_metrics.csv`: 실제 저장 모델의 고정 Test 후속 평가.
- `*_actual_error_cases.csv`, `error_interval_counts.csv`, `error_joint_counts.csv`: 사례 전수와 개발 정상 기준의 구간·조합 조건 건수.
- `rg3_actual_policy_comparison.csv`: 실제 우선검사·무작위 규칙, IF 및 단순 무작위의 같은 예산 비교.
- `rg3_actual_policy_plans.json`: 실제 정책 함수가 생성한 5·10·20% 예시 계획. 현장 승인 정책이 아니다.
- `notebook_execution_audit.json`: 새 셀 실행 및 기존 셀·저장물 보존 확인.
- `report_audit.json`: 보고서에 포함된 그림 전수와 입력 문서 해시.

Test는 이미 관찰한 후속 평가이다. 원본 행은 실제 현장 제품으로 확인되지 않았다.
연구 RF `manual_seven_scenarios_v1`의 그림은 추가 표준화 제거 이전 결과이며 v2 성능이 아니다.
중앙값·IQR 색상은 시각화 표시용이며 RF 입력에 표준화를 추가하지 않았다.
비라벨 분포·점수 변화는 실제 자료를 사용했지만 품질 악화나 재학습 개선을 입증하지 않는다.
재학습 전후 독립 현장 성능 및 자기학습 완료 결과가 없어 해당 항목에는 구현 상태와 절차를 표시하였다.

40개 그림은 결과보고서 본문과 부록 A에 모두 수록되었다. 원본 PNG와 근거 CSV도 위 폴더에 있다.
