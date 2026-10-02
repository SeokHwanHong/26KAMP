# 7단계 전처리 통합 시안 제작 기록

내장 image_gen 도구로 `전체파이프라인_찐막.png`를 편집했습니다. 결과 파일은 `전체파이프라인_7단계.png`입니다. 기존 이미지는 보존합니다.

## 사용 프롬프트

Edit the attached Korean infographic into a SEVEN-stage pipeline, preserving title, clean navy Korean typography, pastel lane design, four independent models IF/OCSVM/LR/RF, existing drift details and readable icons. This is a proposal, not actual implementation. Mandatory primary change: REMOVE ENTIRE old Stage6 검사·라벨 연결 lane with its confirmed/unconfirmed fork, label provenance/timestamps, random inspection note and related connectors. Integrate existing preprocessing of provided labels into Stage2 and Stage4. Old Stage7 모델별 CT becomes Stage6, old Stage8 검증·교체 becomes Stage7. Must have exactly seven numbered lanes and no separate 검사·라벨 연결 lane. Reflow full portrait canvas; no blank gap.
Title unchanged "CN7 · RG3 드리프트 기반 지속 운영·재학습 파이프라인"; subtitle "배치 전처리 → 운영 추론 → 드리프트 판단 → 모델별 CT → 검증·교체". Badge "7단계 개정 기안".
Stage1 데이터 준비: CN7/RG3 분리; 입력24개·제품/배치ID; 라벨 포함 여부·스키마·스케일 확인. Note 현재 생산 시간 순서 미확인.
Stage2 EDA·전처리: four sequential boxes "입력 품질 점검", "원본 ID·제품 빈도 보존", "동일 패턴·위험 이력 집계", "라벨 유무별 자료 분리". Note "라벨 포함: 기존 전처리 적용 / 비라벨: 라벨 생성 없이 보존". Keep max(label) pattern policy and product frequency separate.
Stage3 초기 모델·기준선: IF,OCSVM solid boxes; LR [계획],RF [계획] dashed boxes, no numeric model prefixes, no standalone SVM. Input/model distribution reference [계획]. Note 고정 분할·4-fold / 시간·그룹 검증 추가.
Stage4 배치 유입·추론 [계획]: three boxes "신규 배치: 기존 전처리 재사용" → "저장된 전처리·모델로 추론" → "입력·라벨·점수·예측·버전 저장". Note "원본 제품 빈도 유지 · 추론 시 스케일러 재적합 없음". Another note "비라벨은 정합 확인 후 추론·감지 / 재학습 제외". Arrow from stage2 to stage4 with reuse label if visually clear.
Stage5 드리프트 감지 [계획]: preserve full four cards 값별비율(RG3충전시간2종·사출시간3종), 온도·압력(고정구간비율·범위밖값), 변수조합(사출·충전·압력·온도), 모델반응(IF·OCSVM점수·위험예측비율). Keep window product/pattern counts insufficient waiting; baseline compare → size/persistence → 정상 유지/주의 관찰/검토 원인확인. Note 지속 입력 변화만으로도 검토; output changes strengthen. Note natural variation/window size calibration. No label feedback stage.
Stage6 모델별 CT [보류]: decision "변화·원인 확인 + 라벨 있는 학습 자료 충분?" yes → independent four candidate boxes. IF 확인정상 전체학습[계획]; OCSVM 정상패턴 전체학습[계획]; LR 정상+위험 전체또는점진학습[계획]; RF 정상+위험 전체학습[계획]. No → 관찰·자료누적·학습보류. Note "학습/평가 자료 분리 · 기존/신규 비율·고유 패턴 수 기록". Caution 공정 이상을 새 정상으로 자동 흡수하지 않음. This is operational retraining not initial fit; currently these notebooks are absent so ALL CT marked 계획/보류.
Stage7 검증·교체 [계획]: same evaluation existing/candidate comparison, IF same inspection budget, classification F1/Recall/FPR; pass only relevant model replaced; fail/insufficient preserve. Existing version backup/rollback and postdeployment drift+performance monitor. Feedback right edge to stage4 and stage5 only.
Bottom navy strip "추적 이력: 배치 → 전처리 → 기준선 → 드리프트 판단 → 후보 → 평가 → 운영 버전".
Footer "현재 구현: EDA·패턴 전처리·고정 분할·IF/OCSVM 초기 학습 / 추가 계획: 배치 운영·드리프트·CT·검증/교체·LR/RF".
Do not claim OCSVM operational CT currently implemented. Remove all label inspection linking and random inspection text rather than relocating it. Correct readable Korean; do not include invented stats or thresholds.

## 검수 후 수정

3단계 IF와 OCSVM 사이에 생긴 화살표를 제거합니다. 두 모델은 독립 학습·추론합니다. 나머지 7단계와 전처리 라벨 통합을 유지합니다.
