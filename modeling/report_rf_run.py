"""Write a source-grounded Korean RF report after a completed run."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
import pandas as pd

import rf_pipeline as rf


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def independently_count(frame):
    y, p = frame.label.to_numpy(), frame.prediction.to_numpy()
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0., fpr=fp/(fp+tn) if fp+tn else 0.)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("test_run", type=Path)
    args = parser.parse_args()
    run, tests = args.run.resolve(), args.test_run.resolve()
    overview = read(run / "summary.json")
    test_summary = read(tests / "summary.json")
    assert overview["completed"] and test_summary["passed"]
    batch_checks = read(run / "batch_cli_checks.json")
    assert batch_checks["passed"]
    preservation = read(run / "preservation.json")
    assert preservation["passed"] and preservation["runtime_unchanged"]
    # Recheck preservation at report time, including existing RF results.
    rf.check_protected(read(run / "protected_before.json"))
    logs = list(tests.glob("*.log"))
    count = sum(sum(int(n) for n in re.findall(r"Ran (\d+) tests? in", p.read_text(encoding="utf-8"))) for p in logs)
    provided_scale = all(read(run / dataset / "run_plan.json").get("feature_coordinates") == "provided_v1"
                         for dataset in ("cn7", "rg3"))
    report = ["KAMP 랜덤 포레스트 모델링·테스트 결과 보고서", "",
              f"작성: {datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9))).isoformat()}",
              f"담당: seoloo / 실행 ID: {overview['run_id']}", "상위 기준: 사용자 제공 PNG의 드리프트 기반 7단계 파이프라인", "",
              "1. 이번 작업과 첨부 기록의 구분",
              "팀원 TXT의 실행 이력 보존·학습/평가 분리·모델 감사·수치 오차 검증 방식을 RF에 적용했다.",
              "팀원 기록의 LR·공통 운영 구축 명령을 RF 작업 지시로 실행하지 않았다.",
              "기록에 나온 pipeline_runtime.py·LR 실행기·03.modeling/tests/run_checks.py는 현재 저장소에 없다.",
              "그 기록의 43회 테스트 또는 CT/CD 구현을 이 저장소에서 실행한 결과로 표시하지 않는다.", "",
              "2. 실제 실행 환경", json.dumps(overview["environment"], ensure_ascii=False),
              "과거 RF와 sklearn 1.9.1은 같으나 Python·numpy·pandas 등 환경 전체가 동일하지는 않다.",
              "직접 의존성: modeling/requirements-rf-pipeline.txt", "",
              "3. 모델링 절차",
              "CN7·RG3 독립 실행, 기존 고정 패턴 분할 유지, 정상+위험 train 전체 RF 학습.",
              "데이터셋별 7입력 구성×108설정×4fold = 3,024회 forest 적합(선택 후 최종 적합·진단 재적합 별도).",
              "300개 트리, seed42. 각 설정의 OOF 확률을 임계값 1,001개로 평가, 데이터셋별 총 756,756행.",
              "기존 캐시를 사용하지 않고 처음부터 학습했다. 후보·임계값은 합산 OOF F1로 선택했다.",
              "Test는 선택 후에만 후속 평가하며 임계값을 조정하지 않았다.",
              "4fold OOF는 튜닝 성적이며, 기존 Test도 과거 관찰한 후속 평가 자료다.", "",
              "4. 파생변수",
              "압력 차이(사출/전환), 스크루 RPM 차이, 배압 차이, 배럴 평균·센서 간 산포, 금형 평균·차이가 정의되어 있다.",
              ("값은 제공된 표준화 좌표에서 직접 계산하며 추가 StandardScaler를 사용하지 않는다. PCA의 평균 중심화는 유지한다."
               if provided_scale else "과거 실행의 값은 fold 정상 train의 z 기준이다."),
              "실제 압력 차·RPM 폭·섭씨 온도 차·불량 인과효과로 해석하지 않는다.",
              "활성 센서가 2개 미만이면 해당 항목을 생략한다. PCA는 정상 train에서 공정군별 적합한다.",
              "상수 제거·RF는 정상+위험 train 전체를 사용하며 추론에서 변환을 다시 적합하지 않는다."]
    comparisons = {}
    for dataset in ("cn7", "rg3"):
        folder = run / dataset
        artifact = rf.load_artifact(folder)
        summary, selection = read(folder / "summary.json"), read(folder / "selection_manifest.json")
        comparison = dict(previous_environment=read(rf.ROOT / f"output/random_forest_{dataset}/manual_seven_scenarios_v1/selection_manifest.json"), partitions={})
        for partition in ("oof", "test"):
            name = "selected_oof_predictions.csv" if partition == "oof" else "test_predictions.csv"
            saved = pd.read_csv(folder / name, float_precision="round_trip")
            independently = independently_count(saved)
            for key, value in independently.items():
                assert np.isclose(summary[partition][key], value, rtol=0, atol=1e-15)
            np.testing.assert_array_equal(saved.prediction, saved.probability > summary["threshold"])
            old = pd.read_csv(rf.ROOT / f"output/random_forest_{dataset}/manual_seven_scenarios_v1/{name}", float_precision="round_trip")
            np.testing.assert_array_equal(saved.pattern_row, old.pattern_row)
            comparison["partitions"][partition] = dict(independent_metrics=independently,
                max_probability_difference=float(np.abs(saved.probability.to_numpy()-old.probability.to_numpy()).max()),
                decisions_equal=bool(np.array_equal(saved.prediction, old.prediction)),
                minimum_threshold_margin=float(np.abs(saved.probability.to_numpy()-summary["threshold"]).min()))
        comparison["same_selected_candidate"] = selection["candidate_id"] == comparison["previous_environment"]["candidate_id"]
        comparisons[dataset] = comparison
        report += ["", f"5.{1 if dataset == 'cn7' else 2}. {dataset.upper()} 결과",
                   f"선택: {summary['scenario']}, params={summary['params']}, t={summary['threshold']}",
                   f"개발 OOF: {json.dumps(summary['oof'], ensure_ascii=False)}",
                   f"Test: {json.dumps(summary['test'], ensure_ascii=False)}",
                   f"저장 모델 버전: {artifact['model_version']}",
                   f"과거 결과 비교: {json.dumps(comparison['partitions'], ensure_ascii=False)}",
                   f"과거와 선택 후보 동일: {comparison['same_selected_candidate']}",
                   "입력 구성별 OOF 최상위:", pd.read_csv(folder / "scenario_best.csv")[["scenario", "f1", "threshold", "tp", "fp", "fn"]].to_string(index=False),
                   "파생변수 실제 적합/생략 기록:"]
        for scenario, folds in read(folder / "feature_audit.json").items():
            specs = sorted({s["feature"] for f in folds for s in f["derived_specs"]})
            skipped = sorted({s["feature"] for f in folds for s in f["skipped_specs"]})
            counts = [len(f["feature_names"]) for f in folds]
            report.append(f"- {scenario}: 입력 수 {min(counts)}~{max(counts)}, 생성={specs}, 생략={skipped}")
        report += ["공정군 순열 중요도(고정 임계값, 개발 OOF, 각 20회):",
                   pd.read_csv(folder / "group_permutation_importance.csv").sort_values("f1_drop_mean", ascending=False).to_string(index=False),
                   "중요도는 모델 검증 민감도이며 실제 공정 불량 원인의 증명이 아니다."]
    rf.write_json(run / "independent_comparison.json", comparisons)
    report += ["", "6. 테스트와 보존", f"총 {count}개 명명된 테스트 통과(하위 케이스 별도).",
               "파생변수 수식·적합 범위·분할·엄격한 임계값·저장/재로드·제품 ID/빈도/상충 라벨·잘못된 입력 거부·변조 모델 거부·동일 버전 기준선 비교를 확인했다.",
               "소규모 실제 CN7/RG3 학습과 Windows 병렬 실행을 포함했다. 배치 동작 테스트의 입력은 합성 자료다.",
               "추가로 완료된 CN7/RG3 저장 모델의 실제 CLI에서 합성 배치 각 1회씩 원본 복원·상충 라벨 보존과 정합 미확인 입력 거부를 확인했다.",
               "전체 탐색 결과와 OOF/Test 지표를 파일에서 재검산하고 저장 모델로 Test 예측을 다시 확인했다.",
               json.dumps(preservation, ensure_ascii=False),
               "원본·고정 분할·팀원 소스·기존 RF 결과·runtime 보존. RF 담당 저장 테스트만 단일 CSV 부재를 허용하도록 수정했다.",
               "초기 테스트에서 잘못된 VarianceThreshold 속성 조회를 수정했으며 최종 전체 테스트는 통과했다.",
               "전체 실행 시작 때 Windows 출력 인코딩·병렬 함수 직렬화 오류를 수정하고 새 실행 ID로 재시작했다. 중단 폴더는 완료 결과와 별개로 보존한다.",
               "", "7. PNG 구현 범위와 한계",
               "1·2: 기존 팀원 전처리·분할 검증/재사용.",
               "3: RF 초기 모델·저장/버전/감사·제품 빈도를 보존한 개발 기준선 구현 및 실행.",
               "4: RF 전용 배치 추론 API/CLI와 입력 거부·원본 연결 구현. 현장 신규 배치 미제공, 합성 동작 검증.",
               "5: 고정 값/구간/일부 변수쌍·같은 RF 출력의 변화량 제공. 경보선·윈도 누적·지속성·정상/주의/검토 판단은 미구현.",
               "6: CT는 이미지의 보류 상태 유지. 비라벨 의사 라벨 재학습 없음.",
               "7: 초기 평가·감사만 수행. 운영 승격·롤백 미구현, 실제 운영 포인터 변경 없음.",
               "공정 시간순 미확인, 라벨/비라벨 좌표 정합 미확인. 첨부 팀원 정책 수치를 현장 검증된 기본값으로 복사하지 않았다.",
               "", "8. 성능 해석",
               "Test에서 위험 패턴을 발견하지 못하면 F1·Recall 0이다. 개발 성적이나 기술 테스트 통과로 운영 효과를 주장하지 않는다.",
               "위험 이력 패턴 라벨은 개별 제품 불량 확률과 다르며 정상/불량이 동일 입력으로 관찰된 개별 제품 결과는 현재 입력만으로 구분할 수 없다.",
               "독립 평가 자료·현장 시간·검사량과 오탐 허용 조건 확보가 필요하다. Test를 보고 임계값을 다시 맞추지 않는다.",
               "RG3 OOF에는 임계값과 약 5.55e-17 차이인 확률이 있다. 이번 재현의 확률·판정은 과거와 동일했지만, 적용 시 JSON의 저장 임계값을 그대로 사용하고 화면 표시용 반올림 값을 대신 넣지 않는다.",
               "", "9. 재현 명령", ".\\.work\\rf-env\\Scripts\\python.exe modeling\\run_rf_checks.py",
               ".\\.work\\rf-env\\Scripts\\python.exe modeling\\run_rf_modeling.py",
               f".\\.work\\rf-env\\Scripts\\python.exe modeling\\report_rf_run.py {run} {tests}",
               "전체 흐름: modeling/RF_PIPELINE_WORKFLOW.md · 실행 안내: modeling/RF_TEST_FLOW.md"]
    path = run / "KAMP_RF_모델링_테스트결과보고서_20261003.txt"
    path.write_text("\n".join(report)+"\n", encoding="utf-8")
    # Package RF sources, logs and results; original data/runtime remain in repo.
    sources = ["RF_PIPELINE_WORKFLOW.md", "RF_TEST_FLOW.md", "rf_pipeline.py", "run_rf_checks.py", "run_rf_modeling.py",
               "report_rf_run.py", "probe_rf_batch.py", "test_rf_pipeline.py", "test_rf_threshold_parts.py", "requirements-rf-pipeline.txt", "random_forest_manual.py"]
    archive = run / "KAMP_RF_결과와검증코드_20261003.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(rf.ROOT / "output/random_forest_pipeline/.gitattributes", "results/.gitattributes")
        for name in sources:
            z.write(rf.ROOT / "modeling" / name, f"modeling/{name}")
        for folder in (run, tests):
            for item in folder.rglob("*"):
                if item.is_file() and item != archive:
                    prefix = "results" if folder == run else "tests"
                    z.write(item, f"{prefix}/{item.relative_to(folder).as_posix()}")
    print(f"REPORT={path}")
    print(f"ARCHIVE={archive}")


if __name__ == "__main__":
    main()
