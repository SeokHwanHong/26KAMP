from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT/'tmp/rebuild_cn7_domain_notebook.py'
s = p.read_text(encoding='utf-8')
s = s.replace("archive = ROOT/'modeling/archive/ocsvm_cn7_integrated_ap_20260929.ipynb'", "archive = ROOT/'modeling/archive/ocsvm_cn7_integrated_calibration_20260929.ipynb'")
s = s.replace('output/ocsvm_cn7_integrated/domain_fpr_20260929/', 'output/ocsvm_cn7_integrated/validation_threshold_20260929/')
s = s.replace("OUT = ROOT/'output/ocsvm_cn7_integrated/domain_fpr_20260929'", "OUT = ROOT/'output/ocsvm_cn7_integrated/validation_threshold_20260929'")
s = s.replace('탐색·보정·최종 추론', '탐색·최종 추론')
s = s.replace('동률은 FP → fold 재현율 표준편차 → 입력 수 → 후보 순서로 결정합니다.', '동률은 FP → fold 재현율 표준편차 → 입력 수 → 모델 후보 순서 → |임계값|이 작은 순서 → 높은 임계값으로 결정합니다. 마지막 기준은 동일 성능의 재현 가능한 선택을 위한 규칙입니다.')
start = s.index('- 각 개발 fold의 **정상 학습 행 중 65%')
end = s.index('- 4-fold에서 같은 후보', start)
s = s[:start] + '''- 각 fold의 **정상 train 전체**로 scaler·파생변수 활성 센서·상수 제거·PCA·OCSVM을 적합합니다. 임계값 보정용 추가 분할은 없습니다.
- `nu`, `gamma`와 함께 **숫자 임계값 t**를 탐색합니다. 후보는 `0`과 `±geomspace(1e-6, 1e3, 46)`의 **93개**이며 실행 전 고정합니다. 기본 경계 0과 양·음수의 여러 크기를 포함하는 탐색 범위이며, 물리적 근거나 최적성을 보장하는 범위는 아닙니다. 후보 전체를 설정 파일에 저장합니다.
- `(nu, gamma, t)`마다 **같은 t를 모든 validation fold에 적용**해 `risk_score = -decision_function > t`로 판정합니다. 정상·불량 라벨로 검증 TP·FP·FN·TN을 합산하고 조합을 선택합니다. fold별로 최적 t를 따로 구하거나 평균하지 않습니다.
''' + s[end:]
s = s.replace('- 최종 모델도 개발 정상 행을 동일 비율로 적합/보정 분할합니다. 보정 뒤 전체 정상으로 다시 학습하지 않아 모델과 임계값의 점수 척도를 유지합니다.', '- 최종 모델은 **개발 정상 전체로 재학습**하고 선택된 숫자 t를 그대로 적용합니다. 재학습 모델의 점수 척도 변화로 성능이 달라질 수 있으며 test에서 확인합니다. t를 재보정하지 않습니다.')
s = s.replace("old_root = ROOT/'output/ocsvm_cn7_integrated/expanded_search_20260929'\nprotected_hashes = {str(p.relative_to(ROOT)): workflow.digest(p) for p in old_root.rglob('*') if p.is_file()}", "old_roots = [ROOT/'output/ocsvm_cn7_integrated'/name for name in ['expanded_search_20260929','domain_fpr_20260929']]\nprotected_hashes = {str(p.relative_to(ROOT)): workflow.digest(p) for old_root in old_roots for p in old_root.rglob('*') if p.is_file()}")
s = s.replace("'calibration_fraction': workflow.CALIBRATION_FRACTION", "'threshold_values': workflow.THRESHOLDS")
s = s.replace('각 모델에서 세 오탐 예산을 평가하므로 예산 때문에 모델을 세 번 학습하지 않습니다.', '각 모델에서 93개 t를 비교하므로 **401,016개 (시나리오, nu, gamma, t) 조합**을 평가합니다. 임계값·오탐 예산마다 재학습하지 않습니다.\n`threshold_search.csv`에는 모든 t의 성능을 저장하고 `candidate_search.csv`에는 모델 설정·오탐 조건별 최선 t를 저장합니다. 허용 t가 없으면 최소 오탐 후보를 기록하되 최종 선택에서는 오탐 제약으로 제외합니다.')
s = s.replace(" | set(p['cal'])", '')
s = s.replace("print('후보 설정 수:', expected, '| fold 모델 적합 수:', expected*4)", "assert all(set(p['fit']) == set(Xdev.index[fold_ids.ne(p['fold']) & ydev.eq(0)]) for p in plans)\nprint('후보 설정 수:', expected, '| fold 모델 적합 수:', expected*4)\nprint('임계값 포함 조합 수:', expected*len(workflow.THRESHOLDS))\nprint('fold별 정상 train 전체 행 수:', [len(p['fit']) for p in plans])")
s = s.replace("'feature_count','nu','gamma_multiplier']))", "'feature_count','nu','gamma_multiplier','threshold']))")
s = s.replace('이미 보정한 임계값을 고정하고', 'Validation에서 선택한 임계값을 고정하고')
s = s.replace("assert observed['FPR'] <= choice['budget']+1e-12", "assert observed['FPR'] <= choice['budget']+1e-12\n        assert p.threshold.eq(choice['threshold']).all()")
s = s.replace('최종 적합·보정과 고정 test', '개발 정상 전체 적합과 고정 test')
s = s.replace("{choice['scenario']}, 개발 TP=", "{choice['scenario']}, nu={choice['nu']:.6g}, gamma 배수={choice['gamma_multiplier']:.6g}, t={choice['threshold']:.6g}, 개발 TP=")
s = s.replace("'fit_calibration_validation_disjoint': True", "'fit_validation_disjoint': True, 'all_train_normals_used': True, 'same_selected_threshold_after_refit': True")
s = s.replace("'workflow':'domain_fpr_v1'", "'workflow':'domain_validation_threshold_v2'")
s = s.replace('기존 AP 기반 분석과 실행 결과는', '65:35 분할 이전 실행은 [보정 분할 보존본](archive/ocsvm_cn7_integrated_calibration_20260929.ipynb)에 있습니다. 기존 AP 기반 분석과 실행 결과는')
p.write_text(s, encoding='utf-8')
p = ROOT/'modeling/README.md'
s = p.read_text(encoding='utf-8').replace('output/ocsvm_cn7_integrated/domain_fpr_20260929/', 'output/ocsvm_cn7_integrated/validation_threshold_20260929/')
s = s.replace('정상 적합/보정 분할로 모델별 임계값을 정하고 개발 OOF 오탐률 제약 아래 재현율을 최대화합니다.', '각 fold의 정상 train 전체로 학습하고 validation에서 nu·gamma·숫자 임계값 93개를 함께 비교합니다. 개발 OOF 오탐률 제약 아래 재현율을 최대화합니다. 전체 임계값별 성능은 `threshold_search.csv`에 저장합니다.')
s = s.replace('보정 뒤 모델을 재학습하지 않습니다.', '선택 후 개발 정상 전체로 재학습하고 선택된 숫자 임계값을 그대로 적용합니다. 추가 보정 분할은 없습니다.')
s += '\n이전 65:35 보정 분할 분석은 `archive/ocsvm_cn7_integrated_calibration_20260929.ipynb`, 그 결과는 `output/ocsvm_cn7_integrated/domain_fpr_20260929/`에 보존합니다.\n'
p.write_text(s, encoding='utf-8')
for name in ['tmp/verify_cn7_domain_results.py']:
    p = ROOT/name
    s = p.read_text(encoding='utf-8').replace('domain_fpr_20260929', 'validation_threshold_20260929')
    p.write_text(s, encoding='utf-8')
