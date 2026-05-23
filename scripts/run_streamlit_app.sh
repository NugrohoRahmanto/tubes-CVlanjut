#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

PORT=${PORT:-8501}
BIND_ADDRESS=${BIND_ADDRESS:-127.0.0.1}

exec uv run streamlit run apps/streamlit_app.py \
    --server.port "${PORT}" \
    --server.address "${BIND_ADDRESS}" \
    --server.headless true \
    --server.fileWatcherType none
