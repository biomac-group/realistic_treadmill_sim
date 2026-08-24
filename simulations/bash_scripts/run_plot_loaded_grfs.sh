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

cd "${ROOT_DIR}"
exec "${ROOT_DIR}/.venv/bin/python" -m controller_param_ipopt_fit.plot_loaded_grfs "$@"
