#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_dir}"
exec uv run --no-dev fireredppl-server \
    --config "${repo_dir}/configs/fireredppl_server.yaml" "$@"
