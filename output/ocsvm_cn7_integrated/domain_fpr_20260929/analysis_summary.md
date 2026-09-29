# CN7 도메인·오탐 제약 검증 결과

1%·3%·5%는 비교 조건이며 운영 예산은 확정하지 않았습니다.
선택: 개발 OOF FPR 제약 아래 재현율 최대화. AP는 참고용입니다.

- 1%: legacy_without_cycle, 개발 TP=4, FP=2, 재현율=0.364, FPR=0.004; test TP=0, FP=3, FN=3, TN=116, FPR=0.025
- 3%: legacy_without_cycle, 개발 TP=5, FP=6, 재현율=0.455, FPR=0.013; test TP=0, FP=6, FN=3, TN=113, FPR=0.050
- 5%: pressure, 개발 TP=6, FP=16, 재현율=0.545, FPR=0.034; test TP=0, FP=10, FN=3, TN=109, FPR=0.084

개발 선택 성능은 낙관적일 수 있으며 test도 이전에 관찰한 후속 평가입니다.
모집단 오탐률 보장, 운영 채택, 독립 성능 개선을 주장하지 않습니다.