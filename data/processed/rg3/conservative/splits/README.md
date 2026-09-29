# 고정 test와 개발 4-fold

test 20%는 최종 평가 전용입니다. development 80%에서 cv_fold를 번갈아 검증합니다.
split_assignments.csv의 pattern_row는 상위 X_labeled.csv 기준 0부터 시작하는 행 위치입니다.
fold_indices.json도 같은 원본 패턴 행 위치를 사용하며 X_development의 행 번호가 아닙니다.
development_metadata.csv와 test_metadata.csv는 각 구간 CSV의 행 순서에 대응합니다.
각 fold 학습에서 전처리를 적합하세요. OCSVM은 분할 후 학습 라벨 0만 추출합니다.
비라벨 데이터는 포함하지 않습니다. 모델과 임계값 선택에 test를 사용하지 마세요.
