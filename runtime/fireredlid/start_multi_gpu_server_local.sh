#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  start_multi_gpu_server_local.sh \
    --config PATH \
    --model-dir PATH \
    [--gpu-ids 0,1,2,3,4,5,6,7] \
    [--base-port 12400]

Starts one FireRedLID server process per selected GPU using this repository's
.venv and entrypoint.sh. Run `uv sync --python 3.12` before using it.
EOF
}

config_path=""
model_dir=""
gpu_ids=""
base_port=12400

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            config_path="${2:-}"
            shift 2
            ;;
        --model-dir)
            model_dir="${2:-}"
            shift 2
            ;;
        --gpu-ids)
            gpu_ids="${2:-}"
            shift 2
            ;;
        --base-port)
            base_port="${2:-}"
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${config_path}" || ! -f "${config_path}" ]]; then
    echo "--config must point to an existing YAML file" >&2
    exit 2
fi
if [[ -z "${model_dir}" || ! -d "${model_dir}" ]]; then
    echo "--model-dir must point to an existing directory" >&2
    exit 2
fi
case "${base_port}" in
    "" | *[!0-9]*)
        echo "--base-port must be an integer" >&2
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
entrypoint="${repo_root}/entrypoint.sh"
venv_path="${repo_root}/.venv"
run_dir="${FIREREDLID_RUN_DIR:-/tmp/fireredlid-server-local}"

if [[ ! -x "${entrypoint}" ]]; then
    echo "entrypoint is not executable: ${entrypoint}" >&2
    exit 2
fi
if [[ ! -f "${venv_path}/bin/activate" ]]; then
    echo "missing project environment: ${venv_path}; run uv sync first" >&2
    exit 2
fi

if [[ -z "${gpu_ids}" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is required when --gpu-ids is omitted" >&2
        exit 2
    fi
    gpu_ids="$(
        nvidia-smi --query-gpu=index --format=csv,noheader |
            awk '
                NF {
                    gsub(/[[:space:]]/, "", $0)
                    values = values separator $0
                    separator = ","
                }
                END { print values }
            '
    )"
fi

IFS=',' read -r -a selected_gpus <<< "${gpu_ids}"
if [[ ${#selected_gpus[@]} -eq 0 || -z "${selected_gpus[0]}" ]]; then
    echo "no GPUs selected" >&2
    exit 2
fi
if ((base_port + ${#selected_gpus[@]} - 1 > 65535)); then
    echo "selected ports exceed 65535" >&2
    exit 2
fi

log_dir="${run_dir}/logs"
pid_dir="${run_dir}/pids"
mkdir -p "${log_dir}" "${pid_dir}"

for index in "${!selected_gpus[@]}"; do
    gpu="${selected_gpus[$index]}"
    port=$((base_port + index))
    log_path="${log_dir}/gpu-${gpu}.log"
    pid_path="${pid_dir}/gpu-${gpu}.pid"

    if [[ -f "${pid_path}" ]]; then
        existing_pid="$(cat "${pid_path}")"
        if kill -0 "${existing_pid}" 2>/dev/null; then
            echo "GPU ${gpu} server is already running as PID ${existing_pid}" >&2
            exit 1
        fi
    fi

    nohup env \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        FIREREDLID_VENV_PATH="${venv_path}" \
        "${entrypoint}" \
        --config "${config_path}" \
        --model-dir "${model_dir}" \
        --port "${port}" \
        >"${log_path}" 2>&1 &
    pid=$!
    echo "${pid}" >"${pid_path}"
    echo "GPU ${gpu}: PID ${pid}, http://127.0.0.1:${port}, log=${log_path}"
done
