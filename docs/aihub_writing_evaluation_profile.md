# AI-Hub Korean Writing Evaluation Data Profile

## Scope and method

This profile scans every label JSON directly inside the downloaded ZIP archives. It records aggregate structure, completeness, category, length, score, and split-integrity statistics only; no response text, prompts, feedback, or identifiers are emitted.

## Dataset inventory

| Dataset | Labels (train / validation) | ZIPs | JSON parse errors |
| --- | ---: | ---: | ---: |
| 에세이 글 평가 데이터 | 39,591 / 5,906 | 20 | 0 |
| 서술형 글쓰기 평가 데이터 | 32,006 / 4,000 | 80 | 0 |
| 논술형 글쓰기 평가 데이터 | 16,010 / 2,000 | 48 | 0 |

## 에세이 글 평가 데이터

### Completeness and split integrity

- Required-field missingness: none observed
- Train label/source filename mismatches: 0; validation: 0.
- Candidate-ID overlap across train/validation: 0; normalized-text overlap: 13.
- Question/prompt-group overlap across train/validation: 50 (prompt text is hashed for this check).
- Candidate-ID duplicate rows (train/validation): 0 / 0.
- Normalized-text duplicate rows (train/validation): 71 / 5.

### Categorical composition

- **essay_type** — 글짓기: 5,240, 대안제시: 6,082, 설명글: 16,416, 주장: 10,084, 찬성반대: 7,675
- **essay_level** — 1: 6,460, 2: 26,926, 3: 12,111
- **student_grade_group** — 고등: 10,336, 중등: 17,822, 초등: 17,339

### Length and score distributions

- **essay_len** — n=45,497, min/p05/median/p95/max = 200.00/258.00/555.00/987.00/5033.00, mean=591.91.
- **overall_mean** — n=45,497, mean=25.095, median=26.067, range=0.000–30.000; 10-bin histogram: [0.00, 3.00): 9, [3.00, 6.00): 49, [6.00, 9.00): 110, [9.00, 12.00): 171, [12.00, 15.00): 584, [15.00, 18.00): 1,694, [18.00, 21.00): 3,714, [21.00, 24.00): 7,417, [24.00, 27.00): 14,238, [27.00, 30.00]: 17,511.
- **rater_score** — n=136,491, mean=25.095, median=26.100, range=0.000–30.000; 10-bin histogram: [0.00, 3.00): 209, [3.00, 6.00): 147, [6.00, 9.00): 181, [9.00, 12.00): 1,199, [12.00, 15.00): 2,926, [15.00, 18.00): 6,286, [18.00, 21.00): 12,252, [21.00, 24.00): 21,842, [24.00, 27.00): 33,428, [27.00, 30.00]: 58,021.
- **within_record_rater_sd** — n=45,497, mean=2.000, median=1.764, range=0.000–10.637; 10-bin histogram: [0.00, 1.06): 12,395, [1.06, 2.13): 15,247, [2.13, 3.19): 10,003, [3.19, 4.25): 5,026, [4.25, 5.32): 1,886, [5.32, 6.38): 629, [6.38, 7.45): 206, [7.45, 8.51): 77, [8.51, 9.57): 21, [9.57, 10.64]: 7.

## 서술형 글쓰기 평가 데이터

### Completeness and split integrity

- Required-field missingness: none observed
- Train label/source filename mismatches: 0; validation: 0.
- Candidate-ID overlap across train/validation: 0; normalized-text overlap: 0.
- Question/prompt-group overlap across train/validation: 60 (prompt text is hashed for this check).
- Candidate-ID duplicate rows (train/validation): 0 / 0.
- Normalized-text duplicate rows (train/validation): 0 / 0.

### Categorical composition

- **grade** — 중1: 6,874, 중2: 6,791, 중3: 5,956, 초5: 8,407, 초6: 7,978
- **subject** — 과학: 6,242, 국어: 12,522, 사회: 11,921, 수학: 5,321
- **question_type** — 서술형 글쓰기: 36,006
- **question_level** — 상: 12,000, 중: 12,061, 하: 11,945
- **answer_gender** — 남: 19,104, 여: 16,902
- **answer_region** — 강원: 28, 경기: 8,697, 경남: 588, 경북: 602, 광주: 1,447, 대구: 1,051, 대전: 1,596, 부산: 1,393, 서울: 8,629, 인천: 7,299, 전남: 715, 전북: 817, 제주: 1,789, 충남: 915, 충북: 440

### Length and score distributions

- **text_characters** — n=36,006, min/p05/median/p95/max = 127.00/147.00/283.00/392.00/1315.00, mean=261.55.
- **len_syllable** — n=36,006, min/p05/median/p95/max = 100.00/112.00/214.00/295.00/971.00, mean=198.01.
- **len_word** — n=36,006, min/p05/median/p95/max = 21.00/35.00/68.00/97.00/337.00, mean=63.92.
- **holistic_rater_score** — n=72,012, mean=2.962, median=3.000, range=1.000–4.000; values: 1.0: 1,247, 2.0: 15,498, 3.0: 40,046, 4.0: 15,221.
- **holistic_mean** — n=36,006, mean=2.962, median=3.000, range=1.000–4.000; values: 1.0: 262, 1.5: 656, 2.0: 4,876, 2.5: 5,037, 3.0: 15,532, 3.5: 4,252, 4.0: 5,391.
- **holistic_rater_abs_difference** — n=36,006, mean=0.290, median=0.000, range=0.000–2.000; values: 0.0: 25,807, 1.0: 9,945, 2.0: 254.

## 논술형 글쓰기 평가 데이터

### Completeness and split integrity

- Required-field missingness: none observed
- Train label/source filename mismatches: 0; validation: 0.
- Candidate-ID overlap across train/validation: 0; normalized-text overlap: 0.
- Question/prompt-group overlap across train/validation: 89 (prompt text is hashed for this check).
- Candidate-ID duplicate rows (train/validation): 0 / 0.
- Normalized-text duplicate rows (train/validation): 1 / 0.

### Categorical composition

- **grade** — 중1: 4,325, 중2: 2,411, 중3: 3,237, 초5: 4,466, 초6: 3,571
- **subject** — 과학: 1,865, 국어: 7,328, 사회: 8,817
- **question_type** — 논술형 글쓰기: 18,010
- **question_level** — 상: 4,841, 중: 7,462, 하: 5,707
- **answer_gender** — 남: 9,616, 여: 8,394
- **answer_region** — 강원: 66, 경기: 4,762, 경남: 194, 경북: 207, 광주: 782, 대구: 745, 대전: 835, 부산: 1,119, 서울: 4,097, 세종: 50, 울산: 69, 인천: 3,222, 전남: 311, 전북: 342, 제주: 578, 충남: 410, 충북: 221

### Length and score distributions

- **text_characters** — n=18,010, min/p05/median/p95/max = 765.00/807.00/876.00/1176.55/2812.00, mean=919.28.
- **len_syllable** — n=18,010, min/p05/median/p95/max = 600.00/608.00/655.00/882.00/2116.00, mean=688.97.
- **len_word** — n=18,010, min/p05/median/p95/max = 154.00/188.00/215.00/286.00/697.00, mean=223.10.
- **holistic_rater_score** — n=36,020, mean=2.896, median=3.000, range=1.000–4.000; values: 1.0: 537, 2.0: 7,483, 3.0: 23,200, 4.0: 4,800.
- **holistic_mean** — n=18,010, mean=2.896, median=3.000, range=1.000–4.000; values: 1.0: 107, 1.5: 285, 2.0: 2,209, 2.5: 2,748, 3.0: 9,578, 3.5: 1,474, 4.0: 1,609.
- **holistic_rater_abs_difference** — n=18,010, mean=0.266, median=0.000, range=0.000–2.000; values: 0.0: 13,357, 1.0: 4,507, 2.0: 146.

## Caveats

- Filename pairing and normalized-text checks are ingestion checks, not a semantic leakage audit. Verify IDs and any near-duplicate responses again after creating the modeling table.
- Score meaning and aggregation should be confirmed against AI-Hub's rubric documentation before selecting a training target.
