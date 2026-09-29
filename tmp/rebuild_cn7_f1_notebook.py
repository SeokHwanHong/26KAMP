from pathlib import Path
import hashlib
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'modeling/ocsvm_cn7_integrated.ipynb'
archive = ROOT/'modeling/archive/ocsvm_cn7_integrated_fpr_validation_20260929.ipynb'
previous = nbf.read(archive, as_version=4)
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md('''# CN7 OCSVM — 도메인 가설과 validation F1 기반 탐색

**선택 기준: 4-fold validation의 TP·FP·FN을 합산한 OOF F1 최대화.**

`F1 = 2TP / (2TP + FP + FN)`으로 미탐과 오탐을 함께 반영합니다. 정상 판정 수(TN)가 많다는 이유만으로 점수가 높아지지 않습니다. 별도의 오류 비용 가중치를 두지 않는 기본 목적함수이며 공식 대회 채점 지표나 운영 비용 최적임을 주장하지 않습니다.
FPR·AP·ROC-AUC는 참고 지표입니다. **FPR 상한으로 후보를 제외하거나 FPR로 순위를 정하지 않습니다.**

동률은 fold별 F1 표준편차가 작은 후보 → 입력 수가 적은 후보 순입니다. 여기까지 같으면 모델 후보 순서 → |t|이 작은 값 → 높은 t로 재현 가능하게 결정합니다. 마지막 규칙은 성능 우위를 뜻하지 않습니다.

## A. 학습·검증·선택 규칙

- 기존 패턴 단위 개발/test 분할과 4-fold를 유지합니다. 각 fold의 **정상 train 전체**로 전처리·파생변수·PCA·OCSVM을 적합합니다. 추가 보정 분할은 없습니다.
- 변수 구성과 `nu·gamma·숫자 임계값 t`를 함께 탐색합니다. 각 `(nu, gamma, t)`의 **동일한 t를 네 validation fold에 적용**하고 `-decision_function > t`로 판정합니다.
- 임계값 후보는 사전 고정한 `0`과 `±geomspace(1e-6, 1e3, 46)`, 총 93개입니다. 기본 경계와 여러 크기를 포함하는 탐색 범위이며 범위 자체의 최적성은 보장하지 않습니다.
- 각 fold F1의 단순 평균은 참고용입니다. **선택에는 합산 OOF F1**을 사용합니다. 불량이 fold당 2~3개뿐이므로 fold별 혼동행렬과 F1도 함께 확인합니다.
- 최종 조합을 확정한 뒤 개발 정상 전체로 재학습하고 선택된 t를 그대로 적용합니다. 재학습 후 원시 점수 척도가 달라질 수 있으며 test에서 확인합니다. Test로 t를 다시 고르지 않습니다.
- 여러 후보를 같은 개발 자료로 비교하므로 개발 선택 성능은 낙관적일 수 있습니다. 이전에 본 test의 후속 평가이며 독립 성능 검증은 새 자료가 필요합니다.

구현: [cn7_domain_fpr.py](cn7_domain_fpr.py). 모듈 파일명은 기존 참조를 유지하지만 현재 선택 지표는 F1입니다.
이전 FPR 기준 분석은 [보존본](archive/ocsvm_cn7_integrated_fpr_validation_20260929.ipynb)에 있습니다. 새 결과는 `output/ocsvm_cn7_integrated/validation_f1_20260929/`에 저장합니다.
''')
setup = previous.cells[1].source
setup = setup.replace('validation_threshold_20260929', 'validation_f1_20260929')
setup = setup.replace("# 예산과 후보는 test 평가 전에 고정합니다.\nBUDGETS = workflow.BUDGETS\nprint('오탐 비교 조건:', BUDGETS)", "# 탐색 후보와 F1 선택 규칙은 test 평가 전에 고정합니다.\nprint('선택 지표: 합산 OOF F1')")
setup = setup.replace("['expanded_search_20260929','domain_fpr_20260929']", "['expanded_search_20260929','domain_fpr_20260929','validation_threshold_20260929']")
setup = setup.replace("'budgets': list(BUDGETS), ", "'selection_metric': 'pooled OOF F1', ")
code(setup)
md(previous.cells[2].source)
code(previous.cells[3].source)
md('''## C. F1 기준 공동 탐색

7개 시나리오 × nu 28개 × gamma 배수 22개 × 4-fold = **17,248회 모델 적합**입니다. 각 모델의 validation 점수에 t 93개를 적용하여 **401,016개 조합**을 비교합니다. t만 바꿀 때는 재학습하지 않습니다.
실제 gamma는 배수를 입력 수로 나눕니다. 파생변수는 RBF 거리의 가중치도 바꾸므로 성능 변화가 새 정보 추가만을 의미하지 않습니다.

`threshold_search.csv`: 모든 조합의 합산 F1과 fold F1 표준편차 및 보조 지표.
`candidate_search.csv`: 모델 설정마다 F1 기준 최선 t의 결과.
`top_joint_candidates.csv`: 전체 조합 상위 20개. `top_model_candidates.csv`: 모델 설정별 대표 후보 상위 20개.
''')
code('''search, fold_search, selected, plans = workflow.search_development(Xdev, ydev, fold_ids, OUT)
expected = len(workflow.SCENARIOS)*len(workflow.NU_VALUES)*len(workflow.GAMMA_MULTIPLIERS)
assert len(search) == expected and len(fold_search) == expected*4
assert all(not (set(p['fit']) | set(p['valid'])) & set(test) for p in plans)
assert all(set(p['fit']) == set(Xdev.index[fold_ids.ne(p['fold']) & ydev.eq(0)]) for p in plans)
print('모델 적합:', expected*4, '| 임계값 포함 조합:', expected*len(workflow.THRESHOLDS))
print('fold별 정상 train 전체 행 수:', [len(p['fit']) for p in plans])
display(pd.DataFrame(selected))
''')
md('''## D. 가설군 비교와 F1 선택 근거

모든 가설군·파라미터·t를 동일한 합산 F1 기준으로 비교합니다. FPR은 결과 확인용이며 후보 제외 조건이 아닙니다.
가설군별 대표 후보, 모델 설정별 상위 20개, 전체 조합 상위 20개를 제공합니다. 같은 모델에서 여러 t가 동일 예측을 내면 공동 순위 후보가 생길 수 있습니다.
히트맵은 각 `(nu, gamma)`에서 선택한 t의 합산 F1입니다. 별도로 선택된 모델의 t별 F1·정밀도·재현율을 보여줍니다.
''')
code('''columns = ['scenario','nu','gamma_multiplier','threshold','F1','F1_std','mean_fold_F1',
           'precision','recall','TP','FP','FN','TN','FPR','feature_count']
scenario_comparison = pd.DataFrame([workflow.rank_candidates(part).iloc[0].to_dict()
    for _, part in search.groupby('scenario', sort=False)])
scenario_comparison = workflow.rank_candidates(scenario_comparison)
scenario_comparison.to_csv(OUT/'scenario_comparison.csv', index=False)
print('가설군별 대표 후보'); display(scenario_comparison[columns])
top_models = pd.read_csv(OUT/'top_model_candidates.csv', float_precision='round_trip')
top_joint = pd.read_csv(OUT/'top_joint_candidates.csv', float_precision='round_trip')
print('모델 설정별 상위 20개'); display(top_models[columns])
print('전체 조합 상위 20개'); display(top_joint[columns])
choice = selected[0]
if choice['status'] == 'selected_for_followup':
    part = search.loc[search.scenario.eq(choice['scenario'])]
    matrix = part.pivot(index='nu', columns='gamma_multiplier', values='F1')
    fig, ax = plt.subplots(figsize=(12,7))
    im = ax.imshow(matrix.to_numpy(), origin='lower', aspect='auto', vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(len(matrix.columns)), [f'{v:.3g}' for v in matrix.columns], rotation=60)
    ax.set_yticks(range(len(matrix.index)), [f'{v:.4g}' for v in matrix.index])
    ax.set(xlabel='gamma 배수 / 입력 수', ylabel='nu', title=choice['scenario']+' — 합산 validation F1')
    fig.colorbar(im, ax=ax, label='OOF F1'); fig.tight_layout()
    fig.savefig(OUT/'f1_heatmap.png', dpi=130); plt.close(fig)
    display(Image(filename=str(OUT/'f1_heatmap.png')))
    sensitivity = pd.concat([chunk.loc[chunk.candidate_id.eq(choice['candidate_id'])]
        for chunk in pd.read_csv(OUT/'threshold_search.csv', chunksize=50000, float_precision='round_trip')])
    sensitivity.to_csv(OUT/'selected_threshold_sensitivity.csv', index=False)
    fig, ax = plt.subplots(figsize=(11,4))
    for metric in ['F1','precision','recall']: ax.plot(sensitivity.threshold, sensitivity[metric], label=metric)
    ax.axvline(choice['threshold'], color='black', ls='--', label=f"선택 t={choice['threshold']:.6g}")
    ax.set_xscale('symlog', linthresh=1e-6)
    ticks = [-1e3,-1.,-1e-3,-1e-6,0.,1e-6,1e-3,1.,1e3]
    ax.set_xticks(ticks, [f'{v:g}' for v in ticks])
    ax.minorticks_off()
    ax.set(xlabel='숫자 임계값 t (symlog)', ylabel='validation 지표', ylim=(-.02,1.02), title='선택 모델의 임계값별 판정 성능')
    ax.legend(); fig.tight_layout(); fig.savefig(OUT/'threshold_sensitivity.png', dpi=140); plt.close(fig)
    display(Image(filename=str(OUT/'threshold_sensitivity.png')))
''')
md('''## E. 개발 OOF 재현 및 공정군 중요도 진단

선택한 t를 고정하고 검증 원변수의 공정군을 섞은 뒤 파생변수를 다시 계산합니다. **F1 감소량**을 중심으로 재현율 감소·FPR 증가도 기록합니다. 중요도는 진단용이며 후보 추가·삭제·재선택에 사용하지 않습니다. 변수군 간 상관이 깨지므로 인과 효과로 해석하지 않습니다.
''')
code('''oof_predictions, importance = workflow.development_diagnostics(Xdev, ydev, plans, selected, OUT)
if not importance.empty:
    importance_summary = importance.groupby(['scenario','group'], as_index=False).agg(
        mean_F1_drop=('F1_drop','mean'), mean_recall_drop=('recall_drop','mean'), mean_FPR_increase=('FPR_increase','mean'))
    importance_summary.to_csv(OUT/'group_importance_summary.csv', index=False)
    display(importance_summary)
if choice['status'] == 'selected_for_followup':
    p = oof_predictions
    assert len(p) == len(dev) and p.pattern_row.is_unique
    observed = workflow.classification_metrics(p.y_true, p.y_pred)
    assert all(observed[k] == choice[k] for k in ['TP','FP','FN','TN','F1'])
    assert p.threshold.eq(choice['threshold']).all()
    selected_folds = fold_search.loc[fold_search.candidate_id.eq(choice['candidate_id'])]
    selected_folds.to_csv(OUT/'selected_fold_metrics.csv', index=False)
    display(selected_folds[['fold','threshold','F1','precision','recall','TP','FP','FN','TN','FPR']])
selection_sha = workflow.digest(OUT/'selection_manifest.json')
print('선택 확정 해시:', selection_sha)
''')
md('''## F. 선택 확정 후 개발 정상 전체 재학습과 test 후속 평가

개발 정상 전체로 재학습한 모델에 validation에서 선택한 t를 그대로 적용합니다. 모델 재로딩 후 예측 일치도 확인합니다.
Test의 F1·정밀도·재현율·혼동행렬을 모두 보고하며 결과로 파라미터·변수·t를 바꾸지 않습니다.
''')
code('''test_results = workflow.evaluate_fixed_test(Xdev, ydev, X.loc[test], y.loc[test], selected, OUT)
assert workflow.digest(OUT/'selection_manifest.json') == selection_sha
display(test_results[['status','scenario','threshold','F1','precision','recall','TP','FP','FN','TN','FPR','AP_reference']])
r = test_results.iloc[0]
fig, axes = plt.subplots(1,2, figsize=(10,4))
axes[0].bar(['F1','정밀도','재현율'], [r.F1,r.precision,r.recall], color='steelblue')
axes[0].set(ylim=(0,1), title='고정 test 판정 성능')
cm = np.array([[r.TN,r.FP],[r.FN,r.TP]], dtype=int)
axes[1].imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2): axes[1].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
axes[1].set(xticks=[0,1],yticks=[0,1],xticklabels=['정상','불량'],yticklabels=['정상','불량'],xlabel='예측',ylabel='실제',title='고정 test 혼동행렬')
fig.tight_layout(); fig.savefig(OUT/'test_evaluation.png', dpi=140); plt.close(fig)
display(Image(filename=str(OUT/'test_evaluation.png')))
''')
md('''## G. 실패 유형과 검증 기록

정상·불량 공존 패턴과 불량 전용 패턴을 구분해 탐지 결과를 확인합니다. 이는 후속 진단이며 test에 맞춰 임계값을 수정하지 않습니다.
''')
code('''metadata = pd.read_csv(ROOT/'data/processed/cn7/conservative/labeled_metadata.csv')
predictions = pd.read_csv(OUT/'test_predictions.csv')
joined = predictions.merge(metadata[['pattern_row','pattern_id','pattern_type']], on='pattern_row', validate='many_to_one')
type_results = pd.DataFrame([{'pattern_type':kind,'rows':len(part),
    **workflow.classification_metrics(part.y_true,part.y_pred)} for kind,part in joined.groupby('pattern_type')])
type_results.to_csv(OUT/'test_pattern_type_metrics.csv', index=False)
display(type_results[['pattern_type','rows','F1','TP','FN','FP','TN']])
assert all(workflow.digest(ROOT/p) == sha for p, sha in protected_hashes.items())
workflow.load_data(ROOT)
assert workflow.digest(OUT/'selection_manifest.json') == selection_sha
summary_lines = ['# CN7 validation F1 기반 선택 결과', '',
    '선택: 합산 OOF F1 최대화. 동률은 fold F1 표준편차와 입력 수로 비교.',
    'FPR 상한 없음. FPR·AP는 참고용이며 후보 제외·선택에 사용하지 않습니다.', '']
if choice['status'] == 'selected_for_followup':
    summary_lines += [f"선택: {choice['scenario']}, nu={choice['nu']:.6g}, gamma 배수={choice['gamma_multiplier']:.6g}, t={choice['threshold']:.6g}",
        f"개발 OOF F1={choice['F1']:.6f}, fold F1 표준편차={choice['F1_std']:.6f}, TP={int(choice['TP'])}, FP={int(choice['FP'])}, FN={int(choice['FN'])}"]
else: summary_lines.append('개발 F1이 양수인 후보 없음. test는 경보 없음 기준선.')
summary_lines += [f"Test F1={r.F1:.6f}, TP={r.TP}, FP={r.FP}, FN={r.FN}, TN={r.TN}", '',
    '개발 선택 성능은 낙관적일 수 있고 test도 이전에 관찰한 후속 평가입니다. 독립 성능 개선을 주장하지 않습니다.']
summary = '\\n'.join(summary_lines)
(OUT/'analysis_summary.md').write_text(summary, encoding='utf-8'); print(summary)
workflow.write_json(OUT/'verification.json', {
    'all_grid_fits_converged':True,'grid_fits':expected*4,
    'selection_metric':'pooled OOF F1','FPR_used_for_selection':False,'AP_used_for_selection':False,
    'test_used_for_selection':False,'test_previously_seen':True,'selection_sha256':selection_sha,
    'fit_validation_disjoint':True,'all_train_normals_used':True,'same_selected_threshold_after_refit':True,
    'selected_oof_counts_reproduced':True,'model_reload_predictions_equal':True,
    'old_artifacts_unchanged':True,'protected_artifact_count':len(protected_hashes),
    'source_and_split_hashes_verified':True,
    'artifact_sha256':{p.name:workflow.digest(p) for p in OUT.iterdir() if p.is_file()
        and p.name not in ['verification.json','notebook_execution.json','postrun_audit.json']}})
''')
notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},
    'revision':{'workflow':'domain_validation_f1_v3','previous_notebook':str(archive.relative_to(ROOT)),
                'previous_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}})
nbf.validate(notebook); nbf.write(notebook,path)
print(path)
