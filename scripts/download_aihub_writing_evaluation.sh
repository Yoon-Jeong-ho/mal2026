#!/usr/bin/env bash
# Download the three AI-Hub Korean-writing evaluation datasets requested for this project.
# The API key remains in the repository-root .env file as `AI-HUB=<key>` and is never logged.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
shell_path="$repo_root/tools/aihubshell"
output_root="$repo_root/data/raw/aihub"
log_root="$repo_root/data/logs/aihub"

if [[ ! -x "$shell_path" ]]; then
  echo "Missing executable: $shell_path" >&2
  exit 1
fi

# Accept harmless whitespace around the assignment in a hand-maintained .env file.
api_key="$(sed -n 's/^AI-HUB[[:space:]]*=[[:space:]]*//p' "$repo_root/.env")"
# Permit common quoted or `{key}` notation without passing delimiters to AI-Hub.
case "$api_key" in
  \"*\") api_key="${api_key:1:${#api_key}-2}" ;;
  \'*\') api_key="${api_key:1:${#api_key}-2}" ;;
  \{*\}) api_key="${api_key:1:${#api_key}-2}" ;;
esac
if [[ -z "$api_key" ]]; then
  echo "Missing AI-HUB key in $repo_root/.env" >&2
  exit 1
fi

download_dataset() {
  local dataset_key="$1"
  local directory_name="$2"
  local file_keys="$3"
  local expected_archives="$4"
  local destination="$output_root/$directory_name"
  local log_file="$log_root/${dataset_key}_$(date -u +%Y%m%dT%H%M%SZ).log"
  local existing_archives=0

  mkdir -p "$destination" "$log_root"
  existing_archives="$(find "$destination" -type f -name '*.zip' | wc -l)"
  if [[ "$existing_archives" -eq "$expected_archives" ]]; then
    echo "Dataset $dataset_key already has $expected_archives archives; skipping."
    return 0
  elif [[ "$existing_archives" -ne 0 ]]; then
    echo "Dataset $dataset_key has $existing_archives/$expected_archives archives; refusing to overwrite a partial download." >&2
    return 1
  fi
  (
    cd "$destination"
    "$shell_path" -mode d -datasetkey "$dataset_key" -filekey "$file_keys" -aihubapikey "$api_key"
  ) 2>&1 | tee "$log_file"
  if grep -Fq "Download failed" "$log_file"; then
    echo "AI-Hub rejected dataset $dataset_key; see $log_file" >&2
    return 1
  fi
}

# Dataset 545: requested file keys in 56698--56728 (only published file keys are included).
download_dataset 545 "024_essay_writing_evaluation" \
  "56698,56699,56700,56701,56702,56703,56704,56705,56706,56707,56713,56715,56717,56719,56721,56724,56725,56726,56727,56728" 20

# Dataset 71818: file keys 553407--553486.
download_dataset 71818 "025_descriptive_writing_evaluation" \
  "$(seq -s, 553407 553486)" 80

# Dataset 71819: file keys 553487--553534.
download_dataset 71819 "026_argumentative_writing_evaluation" \
  "$(seq -s, 553487 553534)" 48
