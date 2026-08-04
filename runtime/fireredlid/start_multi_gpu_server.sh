#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  start_multi_gpu_server.sh \
    --config PATH \
    --model-dir PATH \
    [--gpu-ids 0,1,2,3,4,5,6,7] \
    [--instances-per-gpu 1] \
    [--base-port 12400]

Starts one or more FireRedLID server processes per selected GPU. Settings come
from the YAML config; only model_dir and the per-process port are overridden.
EOF
}

config_path=""
model_dir=""
gpu_ids=""
instances_per_gpu=1
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
        --instances-per-gpu)
            instances_per_gpu="${2:-}"
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
case "${instances_per_gpu}" in
    "" | 0* | *[!0-9]*)
        echo "--instances-per-gpu must be a positive integer" >&2
        exit 2
        ;;
esac

entrypoint="/opt/FireRedASR2S/entrypoint.sh"
run_dir="${FIREREDLID_RUN_DIR:-/tmp/fireredlid-server}"

if [[ ! -x "${entrypoint}" ]]; then
    echo "entrypoint is not executable: ${entrypoint}" >&2
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
gpu_count=${#selected_gpus[@]}
process_count=$((gpu_count * instances_per_gpu))
if ((base_port + process_count - 1 > 65535)); then
    echo "selected ports exceed 65535" >&2
    exit 2
fi

log_dir="${run_dir}/logs"
pid_dir="${run_dir}/pids"
mkdir -p "${log_dir}" "${pid_dir}"

for ((instance = 0; instance < instances_per_gpu; instance++)); do
    for index in "${!selected_gpus[@]}"; do
        gpu="${selected_gpus[$index]}"
        port=$((base_port + instance * gpu_count + index))
        file_stem="gpu-${gpu}"
        if ((instance > 0)); then
            file_stem+="-instance-${instance}"
        fi
        log_path="${log_dir}/${file_stem}.log"
        pid_path="${pid_dir}/${file_stem}.pid"

        if [[ -f "${pid_path}" ]]; then
            existing_pid="$(cat "${pid_path}")"
            if kill -0 "${existing_pid}" 2>/dev/null; then
                echo "GPU ${gpu} instance ${instance} is already running as PID ${existing_pid}" >&2
                exit 1
            fi
        fi

        nohup env CUDA_VISIBLE_DEVICES="${gpu}" \
            "${entrypoint}" \
            --config "${config_path}" \
            --model-dir "${model_dir}" \
            --port "${port}" \
            >"${log_path}" 2>&1 &
        pid=$!
        echo "${pid}" >"${pid_path}"
        if ((instances_per_gpu == 1)); then
            echo "GPU ${gpu}: PID ${pid}, http://127.0.0.1:${port}, log=${log_path}"
        else
            echo "GPU ${gpu} instance ${instance}: PID ${pid}, http://127.0.0.1:${port}, log=${log_path}"
        fi
    done
done
