#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/configs/workflow.local.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Configuration file not found: ${CONFIG_PATH}" >&2
  echo "Copy configs/workflow.example.yaml to configs/workflow.local.yaml and set paths.raw_data_dir." >&2
  exit 2
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_workflow.py" --config "${CONFIG_PATH}" "${@:2}"
