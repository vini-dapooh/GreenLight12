#!/usr/bin/env bash
# restart the dev server on :8077 (used during development)
cd "$(dirname "$0")"
for p in $(pgrep -f "app.server:app"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
sleep 1
nohup python3 -W ignore -m uvicorn app.server:app --port "${PORT:-8077}" > /tmp/greenlight_server.log 2>&1 &
sleep 3
curl -s localhost:${PORT:-8077}/api/status; echo
