#!/bin/bash
# run_train.sh
# RunPod Pod에서 enhancement 학습을 실행하고, 로그를 실시간 전송하며, 완료 후 Pod을 종료합니다.
#
# Usage:
#   ./run_train.sh [--keep-pod] [--fold N] <exp_name> [hydra overrides...]
#
# Examples:
#   # Fold 5 학습 (완료 후 pod 자동 종료)
#   ./run_train.sh --fold 5 fold5_dpcrn
#
#   # Pod 유지
#   ./run_train.sh --keep-pod --fold 5 fold5_dpcrn
#
# Prerequisites:
#   - runpodctl 설치 및 인증 완료
#   - "enh-train" 이름의 RUNNING pod 존재 (create_pods.sh로 생성 및 셋업 완료)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/experiments"
REMOTE_PROJECT="/workspace/mfa_enhancement"
POD_NAME="enh-train"
KEEP_POD=false
FOLD_INDEX=""

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-pod) KEEP_POD=true; shift ;;
        --pod-name) POD_NAME="$2"; shift 2 ;;
        --fold) FOLD_INDEX="$2"; shift 2 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) break ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [--keep-pod] [--fold N] <exp_name> [hydra overrides...]"
    exit 1
fi

EXP_NAME="$1"
shift
HYDRA_OVERRIDES=("$@")
LOG_FILE="$SCRIPT_DIR/${EXP_NAME}_train.log"

mkdir -p "$RESULTS_DIR"

# ---- Helper functions ----

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

find_pod_id() {
    local matches
    matches=$(runpodctl get pod 2>/dev/null | awk -v n="$POD_NAME" '$2 == n && /RUNNING/')
    local count
    count=$(echo "$matches" | grep -c . 2>/dev/null || true)
    if [[ "$count" -gt 1 ]]; then
        log "ERROR: Multiple RUNNING pods match '$POD_NAME':"
        echo "$matches" | tee -a "$LOG_FILE"
        log "Use --pod-name to specify an exact pod name."
        exit 1
    fi
    echo "$matches" | head -1 | awk '{print $1}'
}

get_ssh_cmd() {
    local pod_id="$1"
    runpodctl ssh connect "$pod_id" 2>/dev/null | grep -oP 'ssh \S+ -p \d+'
}

remote_exec() {
    local ssh_host="$1"
    local ssh_port="$2"
    shift 2
    ssh $SSH_OPTS -p "$ssh_port" "$ssh_host" "$*"
}

remote_scp_from() {
    local ssh_host="$1"
    local ssh_port="$2"
    local remote_path="$3"
    local local_path="$4"
    scp $SSH_OPTS -P "$ssh_port" "${ssh_host}:${remote_path}" "$local_path"
}

# ---- Main ----

log "=== Enhancement Training: $EXP_NAME ==="
log "Fold: ${FOLD_INDEX:-none}"
log "Hydra overrides: ${HYDRA_OVERRIDES[*]:-none}"
log "Keep pod after training: $KEEP_POD"

# 1. Find running pod and get SSH info
log "Finding RUNNING pod '$POD_NAME'..."
POD_ID=$(find_pod_id)
if [[ -z "$POD_ID" ]]; then
    log "ERROR: No RUNNING pod named '$POD_NAME' found."
    exit 1
fi
log "Found pod: $POD_ID"

SSH_CMD=$(get_ssh_cmd "$POD_ID")
if [[ -z "$SSH_CMD" ]]; then
    log "ERROR: Could not get SSH connection info for pod $POD_ID"
    exit 1
fi

SSH_HOST=$(echo "$SSH_CMD" | awk '{print $2}')
SSH_PORT=$(echo "$SSH_CMD" | awk '{print $4}')
log "SSH: $SSH_HOST port $SSH_PORT"

# 2. Build training command
HYDRA_DIR="./results/experiments/${EXP_NAME}"
TRAIN_CMD="cd $REMOTE_PROJECT && python3 -m src.train hydra.run.dir=$HYDRA_DIR"

if [[ -n "$FOLD_INDEX" ]]; then
    TRAIN_CMD="$TRAIN_CMD cv.enabled=true cv.fold_index=$FOLD_INDEX"
fi

for override in "${HYDRA_OVERRIDES[@]:-}"; do
    if [[ -n "$override" ]]; then
        TRAIN_CMD="$TRAIN_CMD $override"
    fi
done

log "Training command: $TRAIN_CMD"

# 3. Run training via nohup
REMOTE_LOG="/tmp/${EXP_NAME}_stdout.log"
REMOTE_PID_FILE="/tmp/${EXP_NAME}_train.pid"
REMOTE_EXIT_FILE="/tmp/${EXP_NAME}_train.exit"

log "Starting training (nohup)..."
remote_exec "$SSH_HOST" "$SSH_PORT" \
    "rm -f $REMOTE_LOG $REMOTE_PID_FILE $REMOTE_EXIT_FILE"
REMOTE_PID=$(remote_exec "$SSH_HOST" "$SSH_PORT" \
    "nohup bash -c '$TRAIN_CMD > $REMOTE_LOG 2>&1; echo \$? > $REMOTE_EXIT_FILE' </dev/null >/dev/null 2>&1 &
     echo \$! | tee $REMOTE_PID_FILE")
log "Remote training PID: $REMOTE_PID"

# Wait for log to appear
log "Waiting for training to initialize..."
for i in $(seq 1 60); do
    if remote_exec "$SSH_HOST" "$SSH_PORT" "test -f $REMOTE_LOG" 2>/dev/null; then
        break
    fi
    sleep 5
done

# 4. Stream logs via tail -f (reconnects if SSH drops)
log "Streaming training logs..."
LOCAL_LINE_COUNT=0
while true; do
    SKIP=$((LOCAL_LINE_COUNT + 1))
    remote_exec "$SSH_HOST" "$SSH_PORT" "tail -n +${SKIP} --pid=\$(cat $REMOTE_PID_FILE) -f $REMOTE_LOG" 2>&1 | tee -a "$LOG_FILE" || true
    LOCAL_LINE_COUNT=$(wc -l < "$LOG_FILE")

    if remote_exec "$SSH_HOST" "$SSH_PORT" "test -f $REMOTE_EXIT_FILE" 2>/dev/null; then
        TRAIN_EXIT=$(remote_exec "$SSH_HOST" "$SSH_PORT" "cat $REMOTE_EXIT_FILE")
        break
    fi

    log "SSH connection lost, reconnecting in 30s..."
    sleep 30
done

if [[ "$TRAIN_EXIT" -ne 0 ]]; then
    log "ERROR: Training failed with exit code $TRAIN_EXIT"
    log "Pod will NOT be terminated due to training failure."
    exit "$TRAIN_EXIT"
fi

log "Training completed successfully!"

# 5. Transfer results
log "Transferring results from pod..."
LOCAL_EXP_DIR="$RESULTS_DIR/$EXP_NAME"
mkdir -p "$LOCAL_EXP_DIR"

remote_exec "$SSH_HOST" "$SSH_PORT" \
    "cd $REMOTE_PROJECT && tar czf /tmp/${EXP_NAME}_results.tar.gz -C results/experiments $EXP_NAME" 2>&1 | tee -a "$LOG_FILE" || true

remote_scp_from "$SSH_HOST" "$SSH_PORT" "/tmp/${EXP_NAME}_results.tar.gz" "$RESULTS_DIR/${EXP_NAME}_results.tar.gz" 2>&1 | tee -a "$LOG_FILE" || {
    log "WARNING: SCP transfer failed."
}

if [[ -f "$RESULTS_DIR/${EXP_NAME}_results.tar.gz" ]]; then
    tar xzf "$RESULTS_DIR/${EXP_NAME}_results.tar.gz" -C "$RESULTS_DIR/" 2>/dev/null || true
    rm -f "$RESULTS_DIR/${EXP_NAME}_results.tar.gz"
    log "Results transferred to: $LOCAL_EXP_DIR"
else
    log "WARNING: Could not transfer result files. Check pod manually."
fi

# 6. Terminate pod (unless --keep-pod)
if [[ "$KEEP_POD" == "false" ]]; then
    log "Terminating pod $POD_ID..."
    runpodctl remove pod "$POD_ID" 2>&1 | tee -a "$LOG_FILE"
    log "Pod terminated."
else
    log "Pod $POD_ID kept running (--keep-pod)."
fi

log "=== Done: $EXP_NAME ==="
log "Log file: $LOG_FILE"
