# Solar 합의 필터링 파일럿 (2026-07-30)

## 목적과 승인 범위

- 목적: 실제 `train` 글만 원천으로 Solar-Open2-250B가 만든 변형 글을 엄격히
  보존·필터링하고, 동일 Solar의 반복 채점과 서로 다른 두 인코더의
  train-only OOF 예측을 이용해 증강 후보의 신뢰도를 추정한다.
- 사용자는 2026-07-30에 다음 순서를 승인했다: 계속 생성 → 불필요한 후보
  필터링 → `evaluation.txt` 그대로 Solar 채점 → 기존 모델과 일치도 확인 →
  근거가 있을 때만 점수 모델 학습에 사용.
- GPU 승인 범위: 물리 GPU 0--3. GPU 4--7은 조회하거나 사용하지 않았다.
- 시작 Git SHA: `1cbab7ab734afde33bd8bec2f59e900758d045e5`.
- 제한 데이터와 개별 글·생성문·채점 근거는 이 문서나 Git에 기록하지 않는다.

## 고정 프로토콜

- 생성 원천: canonical real `train` 2,000건 중 점수 층화로 고른 20건.
- validation: Solar 생성·채점, OOF 학습, 합의 임계값 보정에 행을 적재하지 않는다.
- 생성 행렬: 20 source × 3 axis × 5 requested score × 8 slots = 2,400회.
- requested score는 편집 지시일 뿐 pseudo-label이나 합의 입력으로 사용하지 않는다.
- Solar 채점 입력: 원문 과제와 후보 글만 입력한다. 생성 목표, 원래 점수,
  candidate metadata는 보이지 않는다.
- 채점 프롬프트: `evaluation.txt` exact section routing,
  SHA-256 `1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145`.
- 반복 채점: temperature 0.1, top-p 0.95. 3개 triplet이 같으면 종료하고,
  다르면 2회 추가한다. 3-of-5 이상의 유일한 triplet만 stable label로 둔다.
- 하드 필터를 첫 judge 이전에 적용하며 모든 거절·불안정·음성 결과를 보존한다.
- Qwen3-Embedding-8B와 KURE는 real train만 5-fold OOF 학습하고 synthetic
  Solar 점수를 학습 target으로 읽지 않는다. 두 인코더 임계값도 real-train OOF
  오차의 사전 고정 80% 분위수(0.25 단위 올림, 최소 0.5)로 보정한다.
- 인코더 epoch 수(각각 3, 5)는 이전 직접 점수 실험에서 고정했으므로 그 선택이
  이전 validation 결과의 영향을 받았을 수 있다. 이번 OOF 임계값 계산 자체에는
  validation 행이나 예측을 넣지 않는다.

## 실행 전 검증과 음성 preflight

초기 pilot 실행은 생성 요청 전에 context audit에서 중단되었다. 1,600 editor
token cap을 적용하면 expression-score-5 요청 4건이 4,096 context를 30 token
초과했다. 출력 디렉터리는 만들어지지 않았으며 이 음성 결과를 보존했다.
과학 프로토콜은 바꾸지 않고 editor cap만 1,570으로 낮춰
`prompt + completion + 128 <= 4,096`을 2,400개 요청 전부에서 확인했다.

CPU 검증:

```text
PYTHONPATH=src .venv-standard/bin/python -m unittest \
  tests.test_solar_consensus_pilot \
  tests.test_solar_target_augmentation \
  tests.test_solar_actual_label_smoke \
  tests.test_solar_target_runner
```

- 결과: 44 tests PASS.
- 추가로 신규 runner 4개를 `py_compile`하고 scoped `git diff --check`를 통과했다.

## Solar 적응형 smoke

- run: `solar-consensus-adaptive-smoke-20260730-001`
- 규모: source 5건, candidate attempt 30건.
- hard-filter valid 18, rejected 12.
- stable modal 17, unstable 1; source modal 5/5 stable.
- strict edit-count compatible 12, relaxed-only 6.
- 이 smoke는 1,600 cap으로 실행되었으며 별도 보존한다. 이후 pilot의 전수
  context audit 때문에 1,570 cap으로 integration-only 조정했다.

## Solar 적응형 pilot 결과

- run: `solar-consensus-adaptive-pilot-20260730-001`
- 실행 시간: 2026-07-30 15:55:50--16:30:22 KST.
- topology: official Solar container, TP4/expert parallel, GPU 0--3,
  endpoint `127.0.0.1:19420`, max inflight 64.
- 2,400/2,400 attempts accounted.
- hard-filter valid 1,280 (53.33%), rejected 1,120 (46.67%).
- stable modal 1,229/1,280 (96.02%), unstable 51.
- modal support: support 3 = 1,052, support 4 = 177.
- strict edit-count compatible 963; relaxed-only 317.
- candidate essay hash 중 중복 초과분 8건이 발견됐다. 중복 hash에 속한 후보는
  이후 consensus core에서 제외하고 disagreement pool에 보존한다.
- source control 20건 중 19건만 stable이어서 pilot의 자동 gate
  `all_source_modal_labels_stable`은 실패했다.
- stable Solar score 분포:

| axis | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| content | 8 | 423 | 620 | 178 | 0 |
| organization | 24 | 563 | 464 | 178 | 0 |
| expression | 23 | 557 | 514 | 130 | 5 |

핵심 음성 결과는 content/organization 5점 후보가 하나도 없고 세 축 모두
2--3점에 집중됐다는 점이다. 따라서 requested score를 label로 사용하는 기존
target augmentation은 허용하지 않으며, 이 pilot 결과만으로 full augmentation
또는 downstream 학습을 승인하지 않는다. OOF 합의와 독립 Qwen3.6 대조까지
완료한 후 사용할 수 있는 범위를 판단한다.

## 산출물

- aggregate:
  `outputs/solar-consensus-pilot-v1/solar-consensus-adaptive-pilot-20260730-001/`
- restricted:
  `data/processed/restricted/solar_consensus_pilot_v1/solar-consensus-adaptive-pilot-20260730-001/`
- stable candidate SHA-256:
  `0d9a69554a3503e857f88fe555bc570a326503a69431e061eedbe7513d0e39bd`
- candidate judge controls SHA-256:
  `c5320787b4f72e445fafa6960584b5ad8ead9c09a0cadc02e019ad345a3b5ad2`
- hard-filter rejected SHA-256:
  `881eb53bdf2d6cf4185517186510de766600ad70a72289e0029d11a4a801219e`

모든 output/restricted 산출물은 ignored 경로에 있고 Git에 포함하지 않는다.

## 후속 단계

1. Qwen3-Embedding-8B/KURE real-train 5-fold OOF 예측.
2. OOF-only 임계값으로 strict·unique consensus core와 disagreement pool 분리.
3. 점수와 무관하게 미리 고정한 각 pool 10%를 Qwen3.6 Q4_K_M로
   `evaluation.txt` exact prompt 재채점.
4. 분포·일치도·실패 gate가 충분할 때만 real-only 대비 증강 혼합 학습을 수행.

## OOF 인코더 실행 결과

- smoke run: `solar-consensus-oof-smoke-20260730-001`.
  - Qwen3-Embedding-8B와 KURE 모두 GPU0에서 one-update train/predict smoke 통과.
- full run: `solar-consensus-oof-20260730-001`.
  - 5 folds × model 2개가 모두 완료됐다.
  - 각 fold는 real train 1,600건으로만 학습하고 held-out real train 400건과
    그 fold source의 stable candidate만 예측했다.
  - candidate 1,229건은 모델별로 정확히 한 fold에서만 예측했다.
  - Qwen은 이전 고정 epoch 3, KURE는 epoch 5를 사용했다.

Real-train OOF aggregate:

| model | content RMSE | organization RMSE | expression RMSE | macro RMSE |
|---|---:|---:|---:|---:|
| Qwen3-Embedding-8B | 0.535967 | 0.717505 | 0.542786 | **0.598753** |
| KURE-v1 | 0.573819 | 0.748719 | 0.575225 | 0.632588 |

두 모델 모두 organization이 가장 어렵고, 목표 0.4213보다 현저히 높다. 이번
OOF 모델의 목적은 synthetic label을 대신 만드는 것이 아니라, 이 알려진 오차를
OOF 임계값으로 반영해 Solar label에 대한 독립적인 불확실성 신호를 만드는 것이다.

## Consensus selection 결과

첫 selection attempt
`solar-consensus-selection-20260730-001`은 aggregate metric의 violation tensor
shape를 잘못 전달해 `TypeError`로 끝났다. 이미 기록된 restricted 중간 산출물은
삭제·덮어쓰기하지 않았다. 중첩 axis cell을 사용하는 최소 synthetic preflight를
추가하고 새 run으로 재실행했다.

- 성공 run: `solar-consensus-selection-20260730-002`.
- OOF calibration threshold:
  - encoder pair gap: 세 축 모두 0.5.
  - Solar vs encoder mean: content 0.75, organization 1.0, expression 0.75.
- consensus core: 328/1,229 (26.69%).
- disagreement pool: 901/1,229 (73.31%).
- duplicate essay hash가 2회 이상인 후보 14건은 전부 core에서 제외했다.
- relaxed edit-count 300건도 core에서 제외했다.
- core에도 1점 및 5점이 전혀 없고 각 축 2--4점만 남았다.

Core score 분포:

| axis | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| content | 0 | 89 | 142 | 97 | 0 |
| organization | 0 | 118 | 113 | 97 | 0 |
| expression | 0 | 110 | 122 | 96 | 0 |

따라서 이 pool은 점수 범위 증강용으로는 부적합하며, 중간 난도 문장의 제한적
regularization 후보로만 볼 수 있다.

## Qwen3.6 독립 대조 결과와 중단 결정

- run: `solar-consensus-qwen36-control-20260730-001`.
- model: pinned Qwen3.6-35B-A3B Q4_K_M GGUF,
  SHA-256 `b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772`.
- topology: GPU 0--3에 독립 llama-server 4개, 서버당 parallel 4.
- prompt: 동일 `evaluation.txt` exact section routing.
- selection 이전에 점수와 무관하게 고정한 control:
  consensus core 33건, disagreement 91건, 총 124건.
- 124/124 JSON 채점 성공.

Solar modal pseudo-label 대비 Qwen3.6:

| stratum | content RMSE | organization RMSE | expression RMSE | macro RMSE | triplet exact |
|---|---:|---:|---:|---:|---:|
| consensus core | 0.627646 | 0.577350 | **0.937437** | 0.714144 | 9.09% |
| disagreement | 0.646206 | 0.564519 | 0.805203 | 0.671976 | 23.08% |
| 전체 | 0.641319 | 0.567962 | 0.842424 | 0.683902 | 19.35% |

Core가 disagreement보다 독립 judge와 더 잘 맞아야 한다는 필요한 방향성이
관찰되지 않았다. 특히 core expression exact agreement는 7/33(21.21%)이고,
Solar 대비 Qwen3.6 평균 편향은 -0.818점이었다. Core의 Solar--encoder mean
expression RMSE는 0.451이지만 Qwen3.6--encoder mean RMSE는 1.257로 커서,
현재 core는 독립적인 품질 합의가 아니라 Solar와 두 유사한 score encoder 사이의
합의를 주로 선택한 것으로 보인다.

결론: 이 결과로 synthetic score를 real train에 혼합하면 judge-specific label
오류를 전파할 위험이 높다. 사전 승인된 fail-closed 규칙에 따라 25/50/100%
혼합 학습과 validation 평가는 실행하지 않았다. 이는 실행 실패가 아니라 현재
증강·필터 설계의 음성 과학 결과다. 다음 실험은 human-calibrated 소량 표본,
독립 judge를 필터 설계에 포함하거나, 1·5점 coverage를 실제로 만드는 생성기를
먼저 검증한 뒤 별도 승인을 받아야 한다.

Qwen3.6 restricted score record SHA-256:
`846e054965d8490a08af174130ee2d684fae8172e37ae87cd04238cb779040db`.
