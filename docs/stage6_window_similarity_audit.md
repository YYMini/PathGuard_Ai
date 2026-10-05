# Stage 6.4 — Causal Trajectory / Window Similarity Identifiability Audit

동일 point-distance 반복 문제는 제거했지만, 강한 route 식별 개선은 확인되지 않았다. W10/gap0의 단기 context와 Train-median 이하 step-distance subset에 제한된 추가 신호가 있다. 긴 window와 큰 gap에서는 분리력이 약해지고, prior-only route memory는 전체 Test에 유효하지 않았다. 이는 identifiability audit이며 detector/threshold 선택 결과가 아니다.

사전 고정한 W={10,25,50}, gap={0,10,25,50}, reference={prior,prefix,combined} **36개 모두**를 보고한다. 아래 gap0 상세 표는 표 길이를 줄이기 위한 공통 단면이다. Test 성능으로 설정을 선택하지 않았고, 전체 grid는 표와 CSV에 유지했다. 기록된 산출물은 로컬 `outputs/`에 있으며 Git 추적 대상이 아니다.

## 요청한 완료 보고 1–38

1. **Stage 6.3 최종 테스트:** 272 PASS, 61.482초. 보호 파일 273개 변경 0. 독립 재검증 64개 query 최대 오차 2.91e-11m.

2. **Stage 6.3 commit SHA:** `8a7172b273d2dd59543ed422845df00a1bad2c7d`, 메시지 `stage6: audit causal context reference`. 검증한 원래 로컬 commit은 `09f27098e3a46eb47f5a36a781775c73296569f9`, 보존 tag는 `stage63-local-verified-09f2709`.

3. **Push:** CLI 인증 실패 후 사용자가 허용한 GitHub connector로 같은 tree를 게시했다. Tree SHA `6e0e0501ba78883f8b256b3d9c125ab020f9a4c0` 일치. 기존 branch `feature/stage-6-route-context`를 fast-forward했으며 force push/PR/main merge 없음.

4. **Local/remote:** fetch 후 양쪽 HEAD 모두 `8a7172b273d2dd59543ed422845df00a1bad2c7d`. Stage 6.4 변경은 미커밋·미푸시다.

5. **W별 availability:** 정상 46,299 및 route 378 endpoint를 설정별로 모두 요청했다. 부족한 current/history window는 삭제하지 않고 status+NA score로 기록했다. gap0의 evaluable/unsupported 수는 다음과 같다.

| window_size | gap | reference_type | normal_evaluable_count | normal_unsupported_count | route_evaluable_count | route_unsupported_count | normal_availability_pct | route_availability_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0 | prior | 42327 | 3972 | 297 | 81 | 91.421 | 78.5714 |
| 10 | 0 | prefix | 45958 | 341 | 378 | 0 | 99.2635 | 100 |
| 10 | 0 | combined | 46108 | 191 | 378 | 0 | 99.5875 | 100 |
| 25 | 0 | prior | 42102 | 4197 | 297 | 81 | 90.935 | 78.5714 |
| 25 | 0 | prefix | 45389 | 910 | 363 | 15 | 98.0345 | 96.0317 |
| 25 | 0 | combined | 45763 | 536 | 363 | 15 | 98.8423 | 96.0317 |
| 50 | 0 | prior | 41728 | 4571 | 297 | 81 | 90.1272 | 78.5714 |
| 50 | 0 | prefix | 44440 | 1859 | 363 | 15 | 95.9848 | 96.0317 |
| 50 | 0 | combined | 45189 | 1110 | 363 | 15 | 97.6025 | 96.0317 |

6. **Prior / Prefix / Combined availability:** 위 표와 아래 전체 36개 config 표를 참조한다. prior에는 point gap을 적용하지 않으므로 네 gap의 결과가 동일하다. no-history / insufficient-history / insufficient-current-window는 상호 배타적 status이며 current 길이 부족을 먼저 표시한다. prior trajectory 수는 각 row에 별도로 저장했다. 충분한 current window 중 prior 없음은 W10/25/50에서 정상 3,821/3,761/3,661행, prior 길이 부족은 실제 dataset에서 0행이다. first trajectory는 첫 window의 관측만으로 prior memory를 가정하지 않는다.

7. **사용자별 availability:** 아래 12개 조합의 combined 범위다. 모든 사용자·36 config의 prior/prefix/combined 상세는 `per_user_window_metrics.csv` 144행에 있다. 사용자별 기록에는 정상/route count, availability, score median, ROC-AUC/AP, hard metrics, 기존 feature와 prefix baseline 상관이 포함된다.

| user_id | normal_avail_min | normal_avail_max | route_avail_min | route_avail_max | auc_min | auc_max |
| --- | --- | --- | --- | --- | --- | --- |
| 001 | 98.0154 | 99.7123 | 100 | 100 | 0.5491 | 0.745 |
| 008 | 95.9929 | 99.4107 | 100 | 100 | 0.5229 | 0.7635 |
| 013 | 98.433 | 99.7746 | 100 | 100 | 0.319 | 0.5151 |
| 017 | 84.2252 | 97.6733 | 87.2881 | 100 | 0.4495 | 0.7506 |

8. **Normal median**, 9. **Route median**, 10. **ROC-AUC**, 11. **AP:** 높은 Frechet score=anomaly로 고정했다. threshold/F1 최적화는 수행하지 않았다. 모든 config의 결과:

| window_size | gap | reference_type | normal_availability_pct | route_availability_pct | normal_score_median | route_score_median | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0 | prior | 91.420981 | 78.571429 | 1850.332816 | 380.477369 | 0.34631 | 0.004973 |
| 10 | 0 | prefix | 99.263483 | 100 | 97.264158 | 173.49815 | 0.668049 | 0.011632 |
| 10 | 0 | combined | 99.587464 | 100 | 85.197101 | 168.393605 | 0.686174 | 0.01227 |
| 10 | 10 | prior | 91.420981 | 78.571429 | 1850.332816 | 380.477369 | 0.34631 | 0.004973 |
| 10 | 10 | prefix | 98.853107 | 99.206349 | 142.867779 | 202.195397 | 0.59164 | 0.009242 |
| 10 | 10 | combined | 99.501069 | 99.206349 | 119.226258 | 189.22642 | 0.6175 | 0.00978 |
| 10 | 25 | prior | 91.420981 | 78.571429 | 1850.332816 | 380.477369 | 0.34631 | 0.004973 |
| 10 | 25 | prefix | 98.239703 | 96.031746 | 205.406921 | 249.325406 | 0.544038 | 0.008153 |
| 10 | 25 | combined | 99.371477 | 96.031746 | 155.19549 | 192.708975 | 0.559666 | 0.008083 |
| 10 | 50 | prior | 91.420981 | 78.571429 | 1850.332816 | 380.477369 | 0.34631 | 0.004973 |
| 10 | 50 | prefix | 97.215923 | 96.031746 | 309.914829 | 293.156778 | 0.518769 | 0.007818 |
| 10 | 50 | combined | 99.155489 | 96.031746 | 203.614552 | 199.590166 | 0.519559 | 0.00736 |
| 25 | 0 | prior | 90.935009 | 78.571429 | 1976.064872 | 420.028191 | 0.324915 | 0.004825 |
| 25 | 0 | prefix | 98.034515 | 96.031746 | 205.02033 | 272.864818 | 0.543966 | 0.008148 |
| 25 | 0 | combined | 98.842308 | 96.031746 | 176.539546 | 200.499814 | 0.547838 | 0.007936 |
| 25 | 10 | prior | 90.935009 | 78.571429 | 1976.064872 | 420.028191 | 0.324915 | 0.004825 |
| 25 | 10 | prefix | 97.626299 | 96.031746 | 252.080679 | 346.609561 | 0.522768 | 0.007974 |
| 25 | 10 | combined | 98.755913 | 96.031746 | 213.550877 | 207.212429 | 0.517813 | 0.007442 |
| 25 | 25 | prior | 90.935009 | 78.571429 | 1976.064872 | 420.028191 | 0.324915 | 0.004825 |
| 25 | 25 | prefix | 97.010735 | 96.031746 | 314.947447 | 360.449903 | 0.505272 | 0.007708 |
| 25 | 25 | combined | 98.62632 | 96.031746 | 254.939941 | 232.493037 | 0.489652 | 0.006989 |
| 25 | 50 | prior | 90.935009 | 78.571429 | 1976.064872 | 420.028191 | 0.324915 | 0.004825 |
| 25 | 50 | prefix | 95.984794 | 96.031746 | 418.88541 | 391.368792 | 0.497515 | 0.007571 |
| 25 | 50 | combined | 98.410333 | 96.031746 | 308.835704 | 266.722558 | 0.47196 | 0.006745 |
| 50 | 0 | prior | 90.127217 | 78.571429 | 2234.53352 | 539.657729 | 0.312037 | 0.004769 |
| 50 | 0 | prefix | 95.984794 | 96.031746 | 382.632276 | 506.251575 | 0.490693 | 0.007484 |
| 50 | 0 | combined | 97.60254 | 96.031746 | 322.117861 | 320.748653 | 0.481697 | 0.007048 |
| 50 | 10 | prior | 90.127217 | 78.571429 | 2234.53352 | 539.657729 | 0.312037 | 0.004769 |
| 50 | 10 | prefix | 95.574418 | 96.031746 | 429.106216 | 506.251575 | 0.491379 | 0.007466 |
| 50 | 10 | combined | 97.516145 | 96.031746 | 355.292913 | 379.059298 | 0.477177 | 0.006934 |
| 50 | 25 | prior | 90.127217 | 78.571429 | 2234.53352 | 539.657729 | 0.312037 | 0.004769 |
| 50 | 25 | prefix | 94.961014 | 94.444444 | 489.057681 | 469.591216 | 0.483205 | 0.007229 |
| 50 | 25 | combined | 97.386553 | 96.031746 | 401.521047 | 391.197863 | 0.469884 | 0.006813 |
| 50 | 50 | prior | 90.127217 | 78.571429 | 2234.53352 | 539.657729 | 0.312037 | 0.004769 |
| 50 | 50 | prefix | 93.937234 | 86.772487 | 584.273609 | 445.70308 | 0.461296 | 0.006425 |
| 50 | 50 | combined | 97.172725 | 96.031746 | 461.278546 | 427.352528 | 0.458567 | 0.006671 |

Normal mean/p75/p90/p95/p99, evaluable/unsupported/status 수는 `window_similarity_summary.csv`의 72행(normal/route × 36)에 저장했다. AP는 각 config의 availability가 바뀌어 class prevalence도 달라진다. 표의 AP를 서로 같은 prevalence로 해석하지 않는다.

12. **Window vs distance_m Pearson/Spearman**, 13. **Window vs Stage 6.3 prefix distance**, 14. **동일 score 비율:** 정상 class의 전체 grid는 아래와 같다. `equal_pct`는 atol=1e-5m, rtol=1e-8의 `isclose` 기준이다. Stage 6.3 prefix 정상 동일률 87.11%에 비해 window 정상 동일률은 0~0.0131%다. 즉 adjacent-point와 수치적으로 같은 score는 제거됐다. 다만 Spearman 상관이 남아 독립적인 route 정보를 입증하지는 않는다. 전체/route class 상관 및 동일률도 `window_config_comparison.csv`에 별도로 저장했다.

| window_size | gap | reference_type | normal_distance_m_pearson | normal_distance_m_spearman | normal_causal_prefix_distance_m_pearson | normal_causal_prefix_distance_m_spearman | equal_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0 | prior | 0.291121 | 0.310153 | 0.438888 | 0.314319 | 0 |
| 10 | 0 | prefix | 0.278394 | 0.793634 | 0.441427 | 0.816861 | 0.013055 |
| 10 | 0 | combined | 0.285292 | 0.757618 | 0.451098 | 0.778613 | 0.013013 |
| 10 | 10 | prior | 0.291121 | 0.310153 | 0.438888 | 0.314319 | 0 |
| 10 | 10 | prefix | 0.292619 | 0.755721 | 0.500129 | 0.794245 | 0.010925 |
| 10 | 10 | combined | 0.294974 | 0.707974 | 0.501234 | 0.740707 | 0.010854 |
| 10 | 25 | prior | 0.291121 | 0.310153 | 0.438888 | 0.314319 | 0 |
| 10 | 25 | prefix | 0.295535 | 0.713948 | 0.529079 | 0.762664 | 0.006596 |
| 10 | 25 | combined | 0.292752 | 0.65656 | 0.520023 | 0.696627 | 0.006521 |
| 10 | 50 | prior | 0.291121 | 0.310153 | 0.438888 | 0.314319 | 0 |
| 10 | 50 | prefix | 0.292475 | 0.681646 | 0.540591 | 0.7369 | 0.002222 |
| 10 | 50 | combined | 0.286079 | 0.611552 | 0.523918 | 0.655798 | 0.002178 |
| 25 | 0 | prior | 0.290951 | 0.321209 | 0.438663 | 0.32605 | 0 |
| 25 | 0 | prefix | 0.281745 | 0.754351 | 0.482019 | 0.778619 | 0 |
| 25 | 0 | combined | 0.283092 | 0.712805 | 0.483121 | 0.734791 | 0 |
| 25 | 10 | prior | 0.290951 | 0.321209 | 0.438663 | 0.32605 | 0 |
| 25 | 10 | prefix | 0.289227 | 0.73836 | 0.510566 | 0.770263 | 0 |
| 25 | 10 | combined | 0.287975 | 0.692192 | 0.505891 | 0.719542 | 0 |
| 25 | 25 | prior | 0.290951 | 0.321209 | 0.438663 | 0.32605 | 0 |
| 25 | 25 | prefix | 0.291909 | 0.718131 | 0.529081 | 0.757346 | 0 |
| 25 | 25 | combined | 0.28837 | 0.664862 | 0.519122 | 0.696901 | 0 |
| 25 | 50 | prior | 0.290951 | 0.321209 | 0.438663 | 0.32605 | 0 |
| 25 | 50 | prefix | 0.290263 | 0.693309 | 0.538417 | 0.74011 | 0 |
| 25 | 50 | combined | 0.284779 | 0.631163 | 0.523713 | 0.667976 | 0 |
| 50 | 0 | prior | 0.29071 | 0.328367 | 0.438619 | 0.332931 | 0 |
| 50 | 0 | prefix | 0.288835 | 0.737767 | 0.515689 | 0.762749 | 0 |
| 50 | 0 | combined | 0.286651 | 0.691775 | 0.509884 | 0.713065 | 0 |
| 50 | 10 | prior | 0.29071 | 0.328367 | 0.438619 | 0.332931 | 0 |
| 50 | 10 | prefix | 0.290266 | 0.729399 | 0.525427 | 0.758487 | 0 |
| 50 | 10 | combined | 0.28726 | 0.680136 | 0.517317 | 0.704262 | 0 |
| 50 | 25 | prior | 0.29071 | 0.328367 | 0.438619 | 0.332931 | 0 |
| 50 | 25 | prefix | 0.290114 | 0.71656 | 0.532725 | 0.750419 | 0 |
| 50 | 25 | combined | 0.286131 | 0.662857 | 0.52189 | 0.689965 | 0 |
| 50 | 50 | prior | 0.29071 | 0.328367 | 0.438619 | 0.332931 | 0 |
| 50 | 50 | prefix | 0.287533 | 0.695655 | 0.536053 | 0.735052 | 0 |
| 50 | 50 | combined | 0.282488 | 0.637068 | 0.522309 | 0.667436 | 0 |

15. **Hard subset:** Stage 6.1 frozen flag `all_8_inside_train_p01_p99`를 그대로 재사용한 **307행**이다. 독립적으로 Train percentile을 다시 계산해 같은 307행을 확인했다. 각 feature의 *marginal* 범위 내라는 뜻이며 joint feature 정보나 anomaly signal이 없다는 증명은 아니다.

16. **Hard availability**, 17. **Hard ROC-AUC/AP**, 18. **Hard route median:** 비교 class는 같은 config의 *전체 evaluable Test 정상*이다. gap0 단면:

| window_size | gap | reference_type | route_total_count | route_evaluable_count | route_availability_pct | route_score_median | roc_auc | average_precision | distance_m_roc_auc | causal_prefix_distance_m_roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0 | prior | 307 | 245 | 79.80456 | 362.471385 | 0.345706 | 0.004106 | 0.797065 | 0.745834 |
| 10 | 0 | prefix | 307 | 307 | 100 | 174.718225 | 0.664769 | 0.009362 | 0.792244 | 0.752915 |
| 10 | 0 | combined | 307 | 307 | 100 | 168.315643 | 0.683646 | 0.009896 | 0.792508 | 0.753248 |
| 25 | 0 | prior | 307 | 245 | 79.80456 | 398.409536 | 0.322682 | 0.003978 | 0.796546 | 0.745259 |
| 25 | 0 | prefix | 307 | 296 | 96.416938 | 221.868918 | 0.536947 | 0.006508 | 0.785678 | 0.744885 |
| 25 | 0 | combined | 307 | 296 | 96.416938 | 196.015202 | 0.540443 | 0.006344 | 0.786472 | 0.74561 |
| 50 | 0 | prior | 307 | 245 | 79.80456 | 533.702689 | 0.309252 | 0.003927 | 0.79564 | 0.744431 |
| 50 | 0 | prefix | 307 | 296 | 96.416938 | 376.361318 | 0.48253 | 0.005982 | 0.78424 | 0.743873 |
| 50 | 0 | combined | 307 | 296 | 96.416938 | 294.922208 | 0.470558 | 0.005594 | 0.785555 | 0.744859 |

36개 전체 hard config 중 ROC-AUC 최대도 0.683646이다(탐색 범위 기술이며 설정 선택 아님). 동일 support에서 W10/gap0 combined의 distance_m ROC-AUC는 0.792508, Stage 6.3 prefix는 0.753248로 window가 더 낮다. p01~p99 hard 범위에서 window의 강한 추가 식별력을 입증하지 못했다.

**Low-distance challenge:** Train 원본 정상만으로 median=11.671237514649514m, p75=26.138368699788003m를 고정했다. 정상과 anomaly 양쪽 모두 같은 distance_m cutoff를 적용했다. 아래는 combined/gap0 단면이고 72개 결과를 `low_distance_challenge_metrics.csv`에 유지했다.

| window_size | gap | reference_type | subset | normal_evaluable_count | route_evaluable_count | roc_auc | average_precision | distance_m_roc_auc | distance_m_average_precision | causal_prefix_distance_m_roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0 | combined | train_median | 32000 | 71 | 0.777043 | 0.007871 | 0.674912 | 0.006801 | 0.661448 |
| 10 | 0 | combined | train_p75 | 41609 | 246 | 0.761699 | 0.01277 | 0.80564 | 0.029891 | 0.772024 |
| 25 | 0 | combined | train_median | 31711 | 71 | 0.629182 | 0.003851 | 0.67603 | 0.006896 | 0.662504 |
| 25 | 0 | combined | train_p75 | 41268 | 241 | 0.623935 | 0.007321 | 0.803056 | 0.028663 | 0.768592 |
| 50 | 0 | combined | train_median | 31222 | 71 | 0.558618 | 0.002817 | 0.677838 | 0.00702 | 0.664452 |
| 50 | 0 | combined | train_p75 | 40703 | 241 | 0.558345 | 0.006303 | 0.802629 | 0.028762 | 0.768444 |

Train-median subset의 W10/gap0 combined는 32,000 normal / 71 route에서 ROC-AUC 0.777043, AP 0.007871이다. 동일 support distance_m 0.674912 및 Stage 6.3 prefix 0.661448보다 높다. 그러나 W25/50와 p75 subset에서는 같은 개선이 확인되지 않는다. 제한된 context 신호의 가설이며 Test에서 최적 W/gap 또는 production cutoff를 선택하는 근거로 사용하지 않는다.

19. **Source pair count:** 설정별 요청 pair 378개. `source_matched_window_delta.csv`에 총 13,608 pair를 기록했다. original 또는 synthetic window가 unsupported이면 delta=NA이며 보간하지 않는다. original lineage는 audit join에만 사용하고 scoring/reference/gate 입력으로 사용하지 않았다.

20. **Delta median**, 21. **Delta>0**, 22. **Delta≥25/50/100m:** gap0 단면. `count`는 양쪽 score가 모두 있는 pair 수이며 전체 요청 pair 수와 구분한다. mean/p25/p75/p90/p95 및 ≥10m도 CSV에 있다.

| window_size | gap | reference_type | total_pair_count | count | mean | median | p25 | p75 | p90 | p95 | positive_pct | ge_10m_pct | ge_25m_pct | ge_50m_pct | ge_100m_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0 | prior | 378 | 297 | 70.211 | 30.7404 | 0 | 147.4206 | 199.9206 | 249.4082 | 63.2997 | 58.2492 | 53.5354 | 45.1178 | 37.3737 |
| 10 | 0 | prefix | 378 | 378 | 62.8457 | 41.4902 | 8.4295 | 115.0372 | 155.294 | 176.9929 | 83.5979 | 73.2804 | 63.4921 | 45.5026 | 31.4815 |
| 10 | 0 | combined | 378 | 378 | 73.3292 | 57.5147 | 16.0105 | 117.2535 | 164.1837 | 204.4766 | 87.5661 | 78.836 | 69.8413 | 52.1164 | 34.6561 |
| 25 | 0 | prior | 378 | 297 | 55.3703 | 12.7206 | 0 | 131.0156 | 167.2952 | 231.1795 | 61.6162 | 53.5354 | 42.7609 | 36.0269 | 30.303 |
| 25 | 0 | prefix | 378 | 363 | 40.2511 | 9.5769 | 0 | 83.3928 | 125.1235 | 171.8985 | 64.4628 | 49.8623 | 41.8733 | 33.0579 | 16.5289 |
| 25 | 0 | combined | 378 | 363 | 49.3797 | 17.3924 | 0 | 89.729 | 171.7338 | 214.6692 | 69.4215 | 54.27 | 46.281 | 35.5372 | 19.8347 |
| 50 | 0 | prior | 378 | 297 | 35.6588 | 0 | -2.827 | 70.8303 | 148.6116 | 231.5464 | 46.4646 | 40.7407 | 33.67 | 28.9562 | 18.8552 |
| 50 | 0 | prefix | 378 | 363 | 26.2289 | 0 | 0 | 42.6884 | 99.3172 | 117.2657 | 44.6281 | 39.6694 | 32.2314 | 22.5895 | 9.9174 |
| 50 | 0 | combined | 378 | 363 | 32.7446 | 0 | 0 | 52.1616 | 119.7218 | 225.4855 | 44.3526 | 40.7713 | 33.6088 | 25.3444 | 15.1515 |

W10 combined median +57.51m, 양수 87.57%라는 paired 변화는 실제다. 그러나 reference까지의 원래 거리 자체가 사용자/이동 구간에 따라 커서 전체 normal-vs-route ranking 개선으로 이어지지 않는다. W50에서는 세 reference 모두 paired delta median=0m다.

23. **사용자별 ROC-AUC/AP**, 24. **사용자 편차:** W10/gap0 단면. 같은 config에서도 user 013 combined ROC-AUC 0.515106이며 다른 세 사용자 0.745~0.764보다 약하다. 사용자별 prevalence가 달라 AP의 절대값을 단순 비교하지 않는다.

| user_id | reference_type | normal_availability_pct | route_availability_pct | normal_score_median | route_score_median | roc_auc | average_precision | hard_roc_auc | hard_average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 001 | prior | 94.204686 | 81.818182 | 2975.515383 | 195.780907 | 0.287038 | 0.003045 | 0.266969 | 0.002162 |
| 008 | prior | 88.426635 | 76.344086 | 453.741924 | 519.389761 | 0.615087 | 0.013145 | 0.621152 | 0.012126 |
| 013 | prior | 92.207792 | 64.556962 | 4724.997025 | 281.784388 | 0.230526 | 0.002012 | 0.232775 | 0.001588 |
| 017 | prior | 74.360168 | 87.288136 | 350.013731 | 445.155931 | 0.565367 | 0.06459 | 0.5558 | 0.055143 |
| 001 | prefix | 99.477424 | 100 | 81.640498 | 167.419866 | 0.716172 | 0.014133 | 0.656522 | 0.005918 |
| 008 | prefix | 98.939305 | 100 | 81.421619 | 156.465909 | 0.733939 | 0.023873 | 0.738014 | 0.02185 |
| 013 | prefix | 99.61361 | 100 | 118.125577 | 135.654296 | 0.501612 | 0.004012 | 0.517196 | 0.003431 |
| 017 | prefix | 95.812006 | 100 | 150.704783 | 205.756416 | 0.723006 | 0.109107 | 0.729607 | 0.096758 |
| 001 | combined | 99.712289 | 100 | 70.438148 | 145.629183 | 0.745006 | 0.016271 | 0.68834 | 0.006552 |
| 008 | combined | 99.410725 | 100 | 70.714526 | 156.465909 | 0.76354 | 0.026939 | 0.768863 | 0.024974 |
| 013 | combined | 99.774606 | 100 | 111.785653 | 135.654296 | 0.515106 | 0.004124 | 0.53004 | 0.003524 |
| 017 | combined | 97.673336 | 100 | 124.054218 | 198.658316 | 0.750612 | 0.10967 | 0.755407 | 0.09625 |

25. **Early/middle/late score:** Stage 6.3의 같은 segment-relative-position bucket을 재사용했다. W10/gap0에서 score median/p90과 paired delta median:

| reference_type | progression_bucket | evaluable_count | score_median | score_p90 | delta_median |
| --- | --- | --- | --- | --- | --- |
| combined | early | 127 | 129.0877 | 218.9743 | 16.971 |
| combined | middle | 118 | 189.6743 | 270.6073 | 94.915 |
| combined | late | 133 | 182.123 | 310.3191 | 98.8135 |
| prefix | early | 127 | 132.8879 | 233.18 | 10.9427 |
| prefix | middle | 118 | 189.6743 | 309.0147 | 82.6901 |
| prefix | late | 133 | 182.123 | 316.7402 | 87.1521 |
| prior | early | 100 | 271.5486 | 3033.0146 | 3.0417 |
| prior | middle | 93 | 430.4171 | 3658.8437 | 101.5406 |
| prior | late | 104 | 432.1113 | 2959.9316 | 63.4898 |

26. **Contamination 신호:** combined median은 early 129.09 → middle 189.67 → late 182.12m로 middle→late는 조금 감소하지만 early보다 높다. 이것만으로 past-memory contamination을 인과적으로 식별할 수 없다. 변화하는 displacement와 window composition이 함께 영향을 준다. 모든 config의 bucket 결과는 108행으로 유지했다. reference overlap 및 gap 규칙으로 인접 point 중복은 막았으나, 관측된 이전 synthetic point 자체는 prefix에 존재할 수 있어 label 기반으로 제거하지 않았다.

27. **Zero-displacement 38행**, 28. **Identical-feature 19행:** W10/gap0의 결과:

| reference_type | boundary_type | total_pair_count | count | median | positive_pct | zero_delta_count |
| --- | --- | --- | --- | --- | --- | --- |
| prior | zero_displacement | 38 | 30 | 0 | 30 | 20 |
| prior | feature_near_identical | 19 | 15 | 0 | 0 | 15 |
| prefix | zero_displacement | 38 | 38 | 0 | 47.3684 | 19 |
| prefix | feature_near_identical | 19 | 19 | 0 | 0 | 19 |
| combined | zero_displacement | 38 | 38 | 1.0512 | 50 | 19 |
| combined | feature_near_identical | 19 | 19 | 0 | 0 | 19 |

W10/gap0 combined의 38 zero-displacement pair 중 19개는 positive delta, 19개는 zero delta이며 median +1.05m다. 현재 point의 displacement가 0이어도 window에 이미 달라진 synthetic history가 들어간 segment 끝에서는 신호가 생길 수 있다. 반면 identical-feature 19행은 이 config에서 전부 delta=0m이며 **모든 config의 available identical pair도 delta=0m**다. config에 따라 prior 15개, prefix 16~19개, combined 18~19개만 available인 점을 표시했다. 동일 관측 window인 시작 경계를 관측만으로 구분할 수는 없다. label은 수정하지 않았다.

29. **신규 테스트:** 35 PASS. 독립 DP oracle, pruning-vs-all-stride-candidates, equal-distance tie, 미래 좌표 perturbation, overlapping-prior rejection, same timestamp rejection, current/self/overlap/gap/width, label/feature mutation, metre projection, availability/NA, source pair/hard/correlation/progression/user aggregate, CSV metadata, checksum 보호와 전체 fixture export를 검증했다.

30. **전체 테스트:** `python -m unittest discover -s tests -v`, **307 PASS**, 92.503초. 기존 272개 유지. 실행은 프로젝트 `.venv`의 Python으로 했다. 테스트 fixture의 기존 학습 테스트도 포함되지만 실제 Stage 5 모델을 재학습하거나 변경하지 않았다.

31. **Checksum:** Stage 5~6.3의 296개 및 별도 초기 Stage 산출물 74개, 총 **370개 변경 0**. 신규 CSV/JSON/PNG checksum도 검증했다. 보호 목록은 `outputs/metrics/stage64_protected_snapshot.json`과 `stage64_early_stages_snapshot.json`이다.

32. **Leakage/causality:** 저장된 **1,680,372행 전체**에서 available reference timestamp, user, trajectory, reference index/time 일치, current 길이, prefix non-overlap/gap, prior trajectory 완료 시각을 독립 검사했다. 위반 0. 288개 고정 audit query의 116,840 reference candidate를 독립 full-grid DP로 비교해 selected reference 및 aligned mean까지 일치, 최대 오차 **8.76e-10m**. ROC-AUC/AP, 상관, hard 36 / low-distance 72 / 사용자 144 / progression 108 group도 독립 재계산했다. split/user/scaler/threshold/feature/label/synthetic 원본 변화 0.

33. **Runtime/candidate/memory:** 실제 분석 **349.04초**, peak process working set **739.91MiB**. 이 수치는 scoring/CSV/figure/checksum까지이며 후속 독립 검증 시간은 별도다. 동일 prior 계산의 gap 재사용을 제외하고 candidate 비교 가능 수는 normal 233,253,668 / route 716,209; endpoint 하한 pruning 후 실제 Frechet DP 호출은 normal 958,895 / route 9,498이다. Combined는 prior+prefix 결과의 최솟값이므로 추가 DP 호출이 아니다. prior gap0+모든 prefix gap의 count로 중복 계산을 제외했다. 상세 count는 `candidate_window_counts.csv`와 JSON에 있다.

34. **Case J/K/L/M/N/O:**

| Case | 판정 | 근거 |
| --- | --- | --- |
| J | 강한 판정 불충족 | 낮은 Pearson만으로 novelty를 주장할 수 없고 hard AUC는 point baseline보다 낮다. Train-median/W10에서 제한된 추가 신호는 있다. |
| K | 일부 지지 | identical adjacent-step score는 제거됐지만 prefix Spearman 0.68~0.79와 전반적인 분리력 저하가 남는다. 단순 수치 복제라는 뜻은 아니다. |
| L | 전체 근거 없음 | prior 전체 ROC-AUC 0.312~0.346, 일부 사용자 외 반복 route-memory 효과가 약하다. |
| M | 제한적 지지 | W10의 짧은 local prefix/combined context가 prior보다 유효하지만 큰 gap/W에서는 약해진다. 장기 route memory 성공은 아니다. |
| N | 지지 | 013의 약한 분리력, 017의 낮은 history/window availability 등 사용자별 차이가 크다. |
| O | 지지, 경계 종류별 구분 | identical 시작 경계는 모든 available config에서 delta=0. zero-displacement 종료 경계 일부는 과거 window 덕분에 구분된다. |

35. **추가 정보 여부:** window는 기존 step-distance와 동일한 값이 아니며 제한된 small-step subset에서 추가 신호를 보인다. 전체 및 307 hard subset의 강한 route 식별 개선이나 conditional feature novelty는 입증하지 못했다. paired delta가 커졌다는 것만으로 route-aware detector 개선을 주장하지 않는다.

36. **Route-aware feature experiment 근거:** broad production feature로 진행할 근거는 부족하다. 별도 Train/Validation 설계의 제한된 가설 실험(A)은 가능하다. Test grid에서 가장 좋은 W/gap을 채택하거나 detector threshold를 조정하지 않았다.

37. **Sequence model 필요성:** 이번 audit가 sequence Autoencoder의 필요성/효과를 증명하지 않는다. 긴 window 단순 matching 실패를 learned temporal model의 성공이나 불필요함으로 확장하지 않는다. label identifiability와 history support를 먼저 정리할 필요가 있다.

38. **Stage 6.5 우선 후보:** **D Synthetic route label refinement의 identifiability 검토**, **E Abstention/cold-start support 설계**, 그 다음 **A 제한된 Window similarity feature experiment**. B personal route memory 및 C sequence model은 후순위 후보이며 이번에 구현하지 않았다. D는 이번 분석에서 label을 수정했다는 뜻이 아니다. Stage 6.5/7, 새 detector/model 학습, commit/push/PR/main merge를 실행하지 않았다.

## 재현 규칙과 출력

- Current: `[t-W+1,...,t]`의 이미 관측된 point만 사용하며 raw trajectory adjacency를 유지한다. 모델용 evaluation row가 아닌 과거 관측점도 window에는 포함한다. 품질/label을 사용해 미래나 anomaly history를 oracle-clean하지 않는다.
- Prior: same Test user의 **trajectory end < current trajectory start**. 앞서 시작했지만 아직 끝나지 않은 trajectory도 제외한다.
- Prefix: `reference_end_index <= current_start_index-gap-1` 및 `reference_max_timestamp < current_min_timestamp`. current/source window overlap=0. 같은 timestamp reference도 fail-fast.
- Score: absolute-space discrete Frechet minimum. Frechet는 sequence order를 유지한 monotone coupling의 maximum point distance 최소값이다. Aligned mean/max는 **Frechet winner**의 보조 값이며 별도 best metric을 고르지 않는다.
- Projection: Train-normal median anchor의 common spherical ECEF→3D ENU rigid transform, radius=6,371,000m. Window별 centering/translation normalization 없음. 세 ENU 축을 유지하므로 구면 chord distance이다. 지표가 arc가 아닌 chord인 한계를 명시하며 100km에서 arc 대비 약 0.00103% 작다. 긴 거리에서 곡률 오차가 커질 수 있다.
- Reference stride=10 고정. stride-1 전체 reference grid의 최솟값을 보장하지 않는다. **제한한 stride-10 grid에서는 정확한 최소값**이다. Endpoint distance lower bound, eps=0 cKDTree, DP safe abandonment로 후보를 줄이며 approximate nearest-neighbor는 사용하지 않는다. Prefix tree는 eligible prefix까지만 들어간다. 모든 Test query endpoint를 계산했다.
- Numba `njit(fastmath=False)`는 DP loop를 가속하며 모델/feature/scaler를 학습하지 않는다. NumPy 기존 설치 버전 2.4.6을 유지했고 Numba 0.68.0/llvmlite 0.50.0을 추가했다. 기술 근거: [Numba supported Python features](https://numba.readthedocs.io/en/stable/reference/pysupported.html).
- Source original, hard flag, boundary flag, progression bucket, labels는 score 이후 audit에만 연결한다. 이전 Test original trajectory는 이미 로그에 존재한다고 가정하며 이 실험 밖의 과거 anomaly memory를 재현한 것은 아니다.
- Row들은 동일 trajectory/segment에서 의존하므로 독립 샘플 confidence interval이나 causal contamination 효과를 주장하지 않는다.

```powershell
python -m unittest discover -s tests -v
python -m src.audit_trajectory_window_similarity --protected-snapshot outputs/metrics/stage64_protected_snapshot.json
```

기존 window audit output directory가 있으면 overwrite를 거부한다. 재실행은 독립 `--output-root`와 해당 root의 보호 snapshot/입력/선행 audit를 준비해야 한다.

| 출력 | 내용 |
| --- | --- |
| `window_similarity_scores.csv` | 1,680,372행: normal/route × 36, selected reference metadata, status 및 NA 유지 |
| `window_similarity_summary.csv` | 정상/route 통계 72행 |
| `window_config_comparison.csv` | 전체 config 36, ROC/AP·상관·동일률 및 동일 support baseline 비교 |
| `per_user_window_metrics.csv` | 4명×36 config, availability/score/분리력/hard/correlation |
| `hard_subset_metrics.csv` | frozen 307-row hard subset의 config 36 |
| `low_distance_challenge_metrics.csv` | Train-median/p75 challenge 72 |
| `source_matched_window_delta.csv` | 378 source pairs×36; unsupported pair도 유지 |
| `source_matched_window_delta_summary.csv` | delta count/mean/quantiles/positive/≥10/25/50/100m |
| `window_progression.csv` | early/middle/late 108 group |
| `boundary_window_audit.csv` / `boundary_window_summary.csv` | boundary row metadata 및 요약 |
| `candidate_window_counts.csv` | candidate eligibility 및 DP 호출 count |
| `window_similarity_audit.json` / `stage6_4_verification.json` | config/정책/시간/메모리/보호/checksum |

Metrics: `outputs/metrics/stage6/window_similarity/`.
Independent checks: `outputs/metrics/stage64_final_checks.json` 및 독립 verifier script/log.

Figures: `outputs/figures/stage6/window_similarity/`:

1. `normal_vs_route_window_score.png`
2. `auc_by_window_size.png`
3. `ap_by_window_size.png`
4. `window_score_vs_distance_m.png`
5. `hard_subset_window_score.png`
6. `per_user_window_auc.png`
7. `anomaly_progression_window_score.png`

모든 figure는 고정 config를 표시한다. scatter의 정상 point만 출력 가독성을 위해 deterministic display sample을 사용했고 metric/scoring query는 thinning하지 않았다.
