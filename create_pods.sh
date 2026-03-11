#!/bin/bash
# GPU 가용성 모니터링 후 자동 Pod 생성 + SSH 대기 + 환경 셋업
#
# Usage:
#   bash create_pods.sh <pod-name> [options]
#
# Options:
#   --gpu <type>      GPU 종류 (기본: "NVIDIA GeForce RTX 5090")
#   --count <N>       생성할 Pod 수 (기본: 1)
#                     N>1이면 이름 뒤에 -1, -2, ... 접미사 자동 부여
#   --no-setup        Pod 생성만 하고 셋업은 스킵
#
# Examples:
#   bash create_pods.sh enh-train
#   bash create_pods.sh enh-train --count 3
#   bash create_pods.sh enh-train --gpu "NVIDIA GeForce RTX 4090" --count 2
#   bash create_pods.sh enh-train --count 3 --no-setup
#
# Known issues (RTX 5090 + PyTorch 2.8 + Ubuntu 24.04):
#   - pip install --break-system-packages 필수 (PEP 668)
#   - datasets 4.x는 torchcodec + FFmpeg ABI 불일치 → datasets<4 고정

set -euo pipefail

REMOTE_PROJECT="/workspace/mfa_enhancement"
REPO_URL="https://github.com/yskim3271/mfa_enhancement.git"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
POD_IMAGE="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
NETWORK_VOLUME_ID="8mdudh5imp"

# ---- Parse arguments ----
if [[ $# -lt 1 ]] || [[ "$1" == -* ]]; then
    echo "Usage: $0 <pod-name> [--gpu <type>] [--count <N>] [--no-setup]"
    exit 1
fi

BASE_NAME="$1"
shift

GPU_TYPE="NVIDIA GeForce RTX 5090"
POD_COUNT=1
RUN_SETUP=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU_TYPE="$2"; shift 2 ;;
        --count) POD_COUNT="$2"; shift 2 ;;
        --no-setup) RUN_SETUP=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Build pod name list ----
POD_NAMES=()
if [[ "$POD_COUNT" -eq 1 ]]; then
    POD_NAMES+=("$BASE_NAME")
else
    for i in $(seq 1 "$POD_COUNT"); do
        POD_NAMES+=("${BASE_NAME}-${i}")
    done
fi

echo "=== Pod Create & Setup ==="
echo "GPU: $GPU_TYPE"
echo "Pods: ${POD_NAMES[*]} (${#POD_NAMES[@]}개)"
echo "Setup: $RUN_SETUP"
echo ""

# ---- Helper functions ----

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

get_pod_id() {
    local name="$1"
    runpodctl get pod 2>/dev/null | awk -v n="$name" '$2 == n && /RUNNING/ {print $1}'
}

get_ssh_info() {
    local pod_id="$1"
    runpodctl ssh connect "$pod_id" 2>/dev/null | grep -oP 'ssh \S+ -p \d+'
}

wait_ssh() {
    local name="$1"
    local max_attempts=30  # 최대 5분 (10초 x 30)
    local pod_id=""

    for attempt in $(seq 1 "$max_attempts"); do
        # pod_id는 한번 찾으면 캐시
        if [[ -z "$pod_id" ]]; then
            pod_id=$(get_pod_id "$name")
            if [[ -z "$pod_id" ]]; then
                log "$name: Pod not RUNNING yet (attempt $attempt/$max_attempts)" >&2
                sleep 10
                continue
            fi
        fi

        local ssh_cmd
        ssh_cmd=$(get_ssh_info "$pod_id" 2>/dev/null || true)
        if [[ -z "$ssh_cmd" ]]; then
            log "$name: SSH info not available yet (attempt $attempt/$max_attempts)" >&2
            sleep 10
            continue
        fi

        local ssh_host ssh_port
        ssh_host=$(echo "$ssh_cmd" | awk '{print $2}')
        ssh_port=$(echo "$ssh_cmd" | awk '{print $4}')

        if ssh $SSH_OPTS -p "$ssh_port" "$ssh_host" "echo ready" &>/dev/null; then
            log "$name: SSH 접속 가능 ($ssh_host:$ssh_port)" >&2
            echo "$ssh_host $ssh_port"
            return 0
        fi

        log "$name: SSH 연결 대기 중 (attempt $attempt/$max_attempts)" >&2
        sleep 10
    done

    log "$name: SSH 접속 시간 초과" >&2
    return 1
}

remote_exec() {
    local ssh_host="$1"
    local ssh_port="$2"
    shift 2
    ssh $SSH_OPTS -p "$ssh_port" "$ssh_host" "$*"
}

run_remote_setup() {
    local name="$1"
    local ssh_host="$2"
    local ssh_port="$3"

    log "$name: 환경 셋업 시작..."

    remote_exec "$ssh_host" "$ssh_port" bash -s <<'SETUP_EOF' || { log "$name: 셋업 실패"; return 1; }
set -euo pipefail

REMOTE_PROJECT="/workspace/mfa_enhancement"
REPO_URL="https://github.com/yskim3271/mfa_enhancement.git"

echo "=== Pod Setup ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo ""

# 1. System packages
if ! command -v ffmpeg &>/dev/null; then
    echo "[1/3] Installing system packages..."
    apt-get update -qq
    apt-get install -y -qq ffmpeg > /dev/null 2>&1
    echo "  ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "[1/3] System packages OK (ffmpeg already installed)"
fi

# 2. Repository
echo "[2/3] Setting up repository..."
if [[ -d "$REMOTE_PROJECT/.git" ]]; then
    echo "  Repository exists, pulling latest..."
    cd "$REMOTE_PROJECT" && git fetch origin && git reset --hard origin/master
else
    echo "  Cloning repository..."
    cd /workspace && git clone "$REPO_URL"
fi
cd "$REMOTE_PROJECT"

# 3. Python dependencies + verification
python3 -c "
import hydra, pesq, datasets, tensorboard
import joblib, scipy, librosa, soundfile
" 2>/dev/null && DEPS_OK=true || DEPS_OK=false

if $DEPS_OK; then
    echo "[3/3] Python dependencies already installed"
else
    echo "[3/3] Installing Python dependencies..."
    pip install --break-system-packages -q -r requirements.txt
fi

python3 -c "
import numpy, torch, torchaudio, hydra, librosa, datasets, scipy, joblib, tensorboard
from pesq import pesq
print(f'  torch: {torch.__version__} (CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)})')
print('All imports OK')
"

echo ""
echo "=== Setup Complete ==="
SETUP_EOF

    log "$name: 셋업 완료"
}

# ---- Phase 1: Create pods ----

create_pod() {
    local name="$1"
    while true; do
        log "$name: Pod 생성 시도..."
        RESULT=$(runpodctl create pod \
            --name "$name" \
            --gpuType "$GPU_TYPE" \
            --gpuCount 1 \
            --imageName "$POD_IMAGE" \
            --networkVolumeId "$NETWORK_VOLUME_ID" \
            --volumePath "/workspace" \
            --containerDiskSize 20 \
            --ports "22/tcp" \
            --startSSH \
            --secureCloud 2>&1) || true

        if echo "$RESULT" | grep -qi "error"; then
            log "$name: 생성 실패 - 60초 후 재시도"
            sleep 60
        else
            log "$name: 생성 성공!"
            echo "$RESULT"
            return 0
        fi
    done
}

log "Phase 1: Pod 생성"
for pod_name in "${POD_NAMES[@]}"; do
    create_pod "$pod_name" &
done
wait
log "모든 Pod 생성 완료"

if [[ "$RUN_SETUP" == "false" ]]; then
    echo ""
    log "=== Done (--no-setup) ==="
    exit 0
fi

# ---- Phase 2: SSH wait + Setup (parallel per pod) ----

setup_pod() {
    local name="$1"
    local ssh_info
    ssh_info=$(wait_ssh "$name") || return 1

    local ssh_host ssh_port
    ssh_host=$(echo "$ssh_info" | awk '{print $1}')
    ssh_port=$(echo "$ssh_info" | awk '{print $2}')

    run_remote_setup "$name" "$ssh_host" "$ssh_port"
}

log ""
log "Phase 2: SSH 접속 대기 + 환경 셋업"
for pod_name in "${POD_NAMES[@]}"; do
    setup_pod "$pod_name" &
done
wait

echo ""
log "=== 완료: ${#POD_NAMES[@]}개 Pod 생성 및 셋업 ==="
