#!/bin/bash
# Self-restarting wrapper for final_orchestrator.py.
#
# Loops the orchestrator until run-logs/RUN_DONE.txt exists. If
# final_orchestrator.py dies prematurely (OOM, system blip, etc.), it
# gets relaunched after a short backoff. This means the user wakes up
# to a finished run regardless of transient failures.
#
# Stops looping when:
#   1. RUN_DONE.txt is created (the orchestrator successfully wrote it).
#   2. We've exceeded MAX_LOOPS (safety bail to prevent infinite restart).

set -u

REPO_ROOT="/Users/sanjayadhikesaven/Downloads/graph"
LOG_DIR="$REPO_ROOT/run-logs"
RUN_DONE="$LOG_DIR/RUN_DONE.txt"
KEEP_ALIVE_LOG="$LOG_DIR/KEEP_ALIVE.log"
PYTHON="/opt/anaconda3/bin/python"
SCRIPT="$REPO_ROOT/final_orchestrator.py"

MAX_LOOPS=20

ts() { date "+%Y-%m-%d %H:%M:%S"; }

log() {
    echo "[$(ts)] $*" >> "$KEEP_ALIVE_LOG"
}

log "KEEP-ALIVE WRAPPER START (pid=$$)"
log "watching for $RUN_DONE"

count=0
while [ $count -lt $MAX_LOOPS ]; do
    if [ -f "$RUN_DONE" ]; then
        log "RUN_DONE.txt found — orchestrator completed. Exiting wrapper."
        exit 0
    fi
    count=$((count + 1))
    log "iteration $count/$MAX_LOOPS — launching $SCRIPT"
    "$PYTHON" "$SCRIPT"
    rc=$?
    log "orchestrator exited rc=$rc"
    if [ -f "$RUN_DONE" ]; then
        log "RUN_DONE.txt found after iteration $count — exiting wrapper."
        exit 0
    fi
    log "RUN_DONE.txt not present; sleeping 30s before relaunch"
    sleep 30
done

log "Reached MAX_LOOPS=$MAX_LOOPS without RUN_DONE.txt; giving up."
exit 1
