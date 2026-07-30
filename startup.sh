#!/usr/bin/env bash
#
# startup.sh — bootstrap + run the Meeting Intelligence stack.
#
#   ./startup.sh              free ports, install anything missing, start both services
#   ./startup.sh --reload     same, but run uvicorn with --reload (slow: reloads ML models)
#   ./startup.sh --backend    backend only
#   ./startup.sh --frontend   frontend only
#   ./startup.sh --clean      wipe venv + node_modules first, then full reinstall
#
# Ports can be overridden: BACKEND_PORT=8001 FRONTEND_PORT=5174 ./startup.sh
#
set -euo pipefail
set -m   # job control: each background job gets its own process group, so we can kill whole trees

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/frontend"
LOG_DIR="$ROOT/logs"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

RELOAD=0
RUN_BACKEND=1
RUN_FRONTEND=1
CLEAN=0

for arg in "$@"; do
  case "$arg" in
    --reload)   RELOAD=1 ;;
    --backend)  RUN_FRONTEND=0 ;;
    --frontend) RUN_BACKEND=0 ;;
    --clean)    CLEAN=1 ;;
    -h|--help)  sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)" >&2; exit 1 ;;
  esac
done

# ---------- pretty output ----------

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- port freeing ----------

free_port() {
  local port="$1" label="$2" pids
  pids="$(lsof -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null || true)"
  [ -z "$pids" ] && { log "port ${port} (${label}) is free"; return; }

  for pid in $pids; do
    local cmd
    cmd="$(ps -o comm= -p "$pid" 2>/dev/null || echo '?')"
    warn "port ${port} (${label}) held by pid ${pid} (${cmd}) — killing"
    kill "$pid" 2>/dev/null || true
  done

  # give them a moment to exit cleanly, then escalate
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 0.3
    lsof -ti "tcp:${port}" -sTCP:LISTEN >/dev/null 2>&1 || { log "port ${port} freed"; return; }
  done

  warn "port ${port} still busy — sending SIGKILL"
  lsof -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null | xargs -r kill -9 2>/dev/null || true
  sleep 0.5
  lsof -ti "tcp:${port}" -sTCP:LISTEN >/dev/null 2>&1 && die "could not free port ${port}"
  log "port ${port} freed"
}

# ---------- prerequisite checks ----------

check_prereqs() {
  command -v python3 >/dev/null || die "python3 not found"
  command -v lsof    >/dev/null || die "lsof not found (sudo apt install lsof)"

  if [ "$RUN_BACKEND" = 1 ]; then
    python3 -c 'import venv' 2>/dev/null || die "python3 venv module missing (sudo apt install python3-venv)"
    command -v ffmpeg >/dev/null || die "ffmpeg not found — Whisper needs it (sudo apt install ffmpeg)"
  fi

  if [ "$RUN_FRONTEND" = 1 ]; then
    command -v node >/dev/null || die "node not found"
    command -v npm  >/dev/null || die "npm not found"
  fi
}

# ---------- setup ----------

setup_backend() {
  local venv="$BACKEND_DIR/venv"
  local stamp="$venv/.deps-installed"

  [ "$CLEAN" = 1 ] && { warn "removing $venv"; rm -rf "$venv"; }

  if [ ! -d "$venv" ]; then
    log "creating backend virtualenv"
    python3 -m venv "$venv"
    "$venv/bin/pip" install --upgrade pip setuptools wheel
  fi

  # reinstall whenever requirements.txt is newer than the last successful install
  if [ ! -f "$stamp" ] || [ "$BACKEND_DIR/requirements.txt" -nt "$stamp" ]; then
    log "installing backend deps (torch + whisper — first run downloads ~2GB, be patient)"
    "$venv/bin/pip" install -r "$BACKEND_DIR/requirements.txt"
    touch "$stamp"
  else
    log "backend deps up to date"
  fi

  setup_database "$venv"
}

# ---------- database ----------
#
# PostgreSQL is a hard dependency now. Fail here with an actionable message
# rather than letting uvicorn boot and every request 500 — at a demo, "the
# database is not running" needs to be obvious in one line.

setup_database() {
  local venv="$1"

  local url
  url="$("$venv/bin/python" -c 'import sys; sys.path.insert(0, "'"$BACKEND_DIR"'"); from config import DATABASE_URL; print(DATABASE_URL)')"

  log "checking database"
  if ! "$venv/bin/python" - <<PY
import sys
sys.path.insert(0, "$BACKEND_DIR")
from sqlalchemy import create_engine, text
from config import DATABASE_URL
try:
    with create_engine(DATABASE_URL, pool_pre_ping=True).connect() as c:
        c.execute(text("SELECT 1"))
except Exception as exc:
    print(exc, file=sys.stderr)
    sys.exit(1)
PY
  then
    warn "cannot connect to PostgreSQL"
    echo
    echo "  Is the service running?    sudo systemctl start postgresql"
    echo "  Does the role/db exist?    sudo -u postgres psql -c \"CREATE ROLE nuance LOGIN PASSWORD 'nuance_dev_pw';\" \\"
    echo "                                                    -c \"CREATE DATABASE nuance OWNER nuance;\""
    echo "  Override the connection:   export DATABASE_URL=postgresql+psycopg://user:pass@host:5432/db"
    echo
    die "database unavailable — refusing to start with a backend that cannot serve a single request"
  fi

  # Migrations are idempotent; running them every boot means a fresh clone and
  # an existing install both end up at the same schema with no manual step.
  log "applying database migrations"
  (cd "$BACKEND_DIR" && "$venv/bin/alembic" upgrade head) \
    || die "alembic upgrade failed — see the output above"
}

setup_frontend() {
  [ "$CLEAN" = 1 ] && { warn "removing $FRONTEND_DIR/node_modules"; rm -rf "$FRONTEND_DIR/node_modules"; }

  if [ ! -f "$FRONTEND_DIR/.env" ] && [ -f "$FRONTEND_DIR/.env.example" ]; then
    log "creating frontend/.env from .env.example"
    sed "s|http://localhost:8000|http://localhost:${BACKEND_PORT}|" \
      "$FRONTEND_DIR/.env.example" > "$FRONTEND_DIR/.env"
  fi

  if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    log "installing frontend deps"
    (cd "$FRONTEND_DIR" && npm install)
  else
    log "frontend deps present"
  fi
}

# ---------- run ----------

PIDS=()
SHUTTING_DOWN=0

shutdown() {
  [ "$SHUTTING_DOWN" = 1 ] && return
  SHUTTING_DOWN=1
  trap - INT TERM
  echo
  log "shutting down"

  # `set -m` makes each background job its own process group leader, so the
  # negative-pid form takes down uvicorn/vite and every child they spawned.
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do
    kill -KILL -- "-$pid" 2>/dev/null || true
  done

  # belt and braces: anything still squatting on our ports goes too
  [ "$RUN_BACKEND"  = 1 ] && lsof -ti "tcp:${BACKEND_PORT}"  -sTCP:LISTEN 2>/dev/null | xargs -r kill -9 2>/dev/null || true
  [ "$RUN_FRONTEND" = 1 ] && lsof -ti "tcp:${FRONTEND_PORT}" -sTCP:LISTEN 2>/dev/null | xargs -r kill -9 2>/dev/null || true

  wait 2>/dev/null || true
  log "stopped"
  exit 0
}
trap shutdown INT TERM

start_backend() {
  local venv="$BACKEND_DIR/venv"
  local args=(--host 0.0.0.0 --port "$BACKEND_PORT")
  [ "$RELOAD" = 1 ] && args+=(--reload)

  log "starting backend on http://localhost:${BACKEND_PORT}  (log: logs/backend.log)"
  (
    cd "$BACKEND_DIR"
    exec "$venv/bin/uvicorn" main:app "${args[@]}" >"$LOG_DIR/backend.log" 2>&1
  ) &
  PIDS+=("$!")
}

start_frontend() {
  log "starting frontend on http://localhost:${FRONTEND_PORT}  (log: logs/frontend.log)"
  (
    cd "$FRONTEND_DIR"
    exec npm run dev -- --port "$FRONTEND_PORT" --strictPort >"$LOG_DIR/frontend.log" 2>&1
  ) &
  PIDS+=("$!")
}

# ---------- main ----------

mkdir -p "$LOG_DIR"
check_prereqs

[ "$RUN_BACKEND"  = 1 ] && free_port "$BACKEND_PORT"  "backend"
[ "$RUN_FRONTEND" = 1 ] && free_port "$FRONTEND_PORT" "frontend"

[ "$RUN_BACKEND"  = 1 ] && setup_backend
[ "$RUN_FRONTEND" = 1 ] && setup_frontend

[ "$RUN_BACKEND"  = 1 ] && start_backend
[ "$RUN_FRONTEND" = 1 ] && start_frontend

echo
log "up and running — Ctrl-C to stop both"
[ "$RUN_BACKEND"  = 1 ] && echo "    backend   http://localhost:${BACKEND_PORT}       (docs: /docs)"
[ "$RUN_FRONTEND" = 1 ] && echo "    frontend  http://localhost:${FRONTEND_PORT}"
echo "    logs      tail -f logs/*.log"
echo

# if either service dies, tear the other one down instead of hanging forever
wait -n || true
warn "a service exited — check logs/ for why"
shutdown
