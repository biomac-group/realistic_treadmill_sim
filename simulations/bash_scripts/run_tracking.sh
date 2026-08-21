#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HOME="${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/mplconfig"
if [[ -n "${IPOPT_LIBRARY:-}" ]]; then
    export LD_PRELOAD="${IPOPT_LIBRARY}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

"${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/patch_biosym_0_1_8_py311.py"
exec "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/script_tracking.py" "$@"
