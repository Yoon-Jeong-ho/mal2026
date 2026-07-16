# AI-Hub Korean Writing Evaluation Data

## Acquisition

- Source: AI-Hub `aihubshell` API guide, accessed 2026-07-16.
- Downloader: `scripts/download_aihub_writing_evaluation.sh`.
- API key: repository-root `.env`, key name `AI-HUB` (not tracked or documented here). The downloader tolerates whitespace around the assignment and quoted or brace-delimited values, but never logs the key.
- Raw download root: `data/raw/aihub/`.
- Logs: `data/logs/aihub/`.
- The downloader deliberately selects only the requested published file keys; it does not request supplemental education material. It skips a complete download and refuses to overwrite a partial one.

AI-Hub requires the account's download approval for each dataset. Its CLI downloads selected files, merges multipart archives, and leaves the delivered directory structure under the directory from which it is invoked. Confirm the archive contents and split assignments before preprocessing.

## Requested datasets

| AI-Hub dataset key | Dataset | Requested file keys | Target population / prompts | Intended role |
| --- | --- | --- | --- | --- |
| 545 | `024.에세이 글 평가 데이터` | Published keys within 56698--56728 | Five task types: creative composition, proposing alternatives, expository writing, assertion, and pro/con. | General essay-quality evaluation. |
| 71818 | `25.서술형 글쓰기 평가 데이터` | 553407--553486 | Grade 5 through middle-school year 3; Korean, science, social studies, and mathematics written responses. | Constructed-response evaluation. |
| 71819 | `26.논술형 글쓰기 평가 데이터` | 553487--553534 | Grade 5 through middle-school year 3; Korean, science, and social studies essays. | Argumentative/essay evaluation. |

## Observed delivery (2026-07-16)

The download completed successfully with `aihubshell` v0.6 (SHA-256 `3475a89b89ca10cdebfd7ef0542ec54650759bd5c15491e4dc0da6c15d93390e`). Every delivered ZIP passed Python `zipfile.testzip()` CRC validation. Individual archive checksums are in `data/manifests/aihub_writing_evaluation_20260716.sha256`.

| Dataset | ZIPs / compressed size | Train labels | Validation labels | Source format / label format |
| --- | ---: | ---: | ---: | --- |
| Essay (545) | 20 / 139,345,684 B | 39,591 JSON files | 5,906 JSON files | JSON / JSON |
| Descriptive (71818) | 80 / 143,708,684 B | 32,006 JSON files | 4,000 JSON files | CSV / JSON |
| Argumentative (71819) | 48 / 100,269,199 B | 16,010 JSON files | 2,000 JSON files | CSV / JSON |

The source and label file counts match within every split. This supports one-to-one pairing, but a preprocessing step must validate identifiers before treating those counts as unique examples. Observed label schemas are:

- Essay: `paragraph`, `score`, `student`, `rubric`, `correction`, and `info`.
- Descriptive and argumentative: `essay_question`, `essay_answer`, `expert`, `rubric`, and `score`.

## Delivered organization

Each dataset contains separate `Training` and `Validation` branches, and each branch has source (`TS`/`VS`) and labeling (`TL`/`VL`) zip files. The labels must be joined to their matching source partition; validation must remain isolated from training and model-selection steps.

## Reproducibility record

Record the actual download time, downloader SHA-256, file inventory, extracted record counts, schema, and any failed/retried files after acquisition. Do not place API keys or individual writing content in this document.
