#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  configure_nginx.sh \
    --instances 4|6|8|16 \
    [--base-port 12400] \
    [--listen-port 12345]

Writes the FireRedLID Nginx upstream configuration, validates it, then starts
Nginx or reloads the running master process.
EOF
}

instances=""
base_port=12400
listen_port=12345

while [[ $# -gt 0 ]]; do
    case "$1" in
        --instances)
            instances="${2:-}"
            shift 2
            ;;
        --base-port)
            base_port="${2:-}"
            shift 2
            ;;
        --listen-port)
            listen_port="${2:-}"
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

if [[ "${instances}" != "4" && "${instances}" != "6" && "${instances}" != "8" && "${instances}" != "16" ]]; then
    echo "--instances must be 4, 6, 8, or 16" >&2
    exit 2
fi
for port_name in base_port listen_port; do
    port="${!port_name}"
    case "${port}" in
        "" | 0* | *[!0-9]*)
            echo "--${port_name//_/-} must be an integer between 1 and 65535" >&2
            exit 2
            ;;
    esac
    if ((port > 65535)); then
        echo "--${port_name//_/-} must be an integer between 1 and 65535" >&2
        exit 2
    fi
done
if ((base_port + instances - 1 > 65535)); then
    echo "upstream ports exceed 65535" >&2
    exit 2
fi

nginx_bin="${NGINX_BIN:-nginx}"
config_path="${FIREREDLID_NGINX_CONFIG:-/etc/nginx/conf.d/fireredlid.conf}"
pid_file="${NGINX_PID_FILE:-/run/nginx.pid}"

if ! command -v "${nginx_bin}" >/dev/null 2>&1; then
    echo "nginx executable not found: ${nginx_bin}" >&2
    exit 1
fi

config_dir="$(dirname "${config_path}")"
mkdir -p "${config_dir}"
generated_config="$(mktemp "${config_dir}/.fireredlid.conf.XXXXXX")"
backup_config=""

cleanup() {
    rm -f "${generated_config}"
    if [[ -n "${backup_config}" ]]; then
        rm -f "${backup_config}"
    fi
}
trap cleanup EXIT

{
    cat <<EOF
upstream fireredlid_servers {
    zone fireredlid_servers 256k;
    least_conn;
    keepalive 64;
EOF
    for ((index = 0; index < instances; index++)); do
        port=$((base_port + index))
        echo "    server 127.0.0.1:${port} max_fails=2 fail_timeout=10s;"
    done
    cat <<EOF
}

server {
    listen ${listen_port};
    server_name _;

    client_max_body_size 192m;
    client_body_timeout 300s;

    location / {
        proxy_pass http://fireredlid_servers;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 5s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        proxy_request_buffering off;
    }
}
EOF
} >"${generated_config}"

if [[ -f "${config_path}" ]]; then
    backup_config="$(mktemp "${config_dir}/.fireredlid.backup.XXXXXX")"
    cp "${config_path}" "${backup_config}"
fi
install -m 0644 "${generated_config}" "${config_path}"

if ! "${nginx_bin}" -t; then
    if [[ -n "${backup_config}" ]]; then
        install -m 0644 "${backup_config}" "${config_path}"
    else
        rm -f "${config_path}"
    fi
    echo "Nginx validation failed; restored the previous configuration" >&2
    exit 1
fi

nginx_pid=""
if [[ -s "${pid_file}" ]]; then
    nginx_pid="$(cat "${pid_file}")"
fi
if [[ "${nginx_pid}" =~ ^[0-9]+$ ]] && kill -0 "${nginx_pid}" 2>/dev/null; then
    "${nginx_bin}" -s reload
    action="reloaded"
else
    "${nginx_bin}"
    action="started"
fi

echo "Nginx ${action}: http://0.0.0.0:${listen_port} -> 127.0.0.1:${base_port}-$((base_port + instances - 1))"
echo "Configuration: ${config_path}"
