#!/bin/bash
# RTX 5090 가용성 모니터링 후 자동 Pod 생성

RUNPOD_API_KEY=$(grep -oP 'apikey = "\K[^"]+' ~/.runpod/config.toml)

while true; do
    echo "[$(date '+%H:%M:%S')] RTX 5090 Pod 생성 시도..."
    RESULT=$(runpodctl create pod \
        --name "enh-train-1" \
        --gpuType "NVIDIA GeForce RTX 5090" \
        --gpuCount 1 \
        --imageName "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404" \
        --networkVolumeId "8mdudh5imp" \
        --volumePath "/workspace" \
        --containerDiskSize 20 \
        --ports "22/tcp" \
        --startSSH \
        --secureCloud 2>&1)

    if echo "$RESULT" | grep -qi "error"; then
        echo "[$(date '+%H:%M:%S')] 실패: $RESULT - 60초 후 재시도"
    else
        echo "[$(date '+%H:%M:%S')] Pod 생성 성공!"
        echo "$RESULT"
        break
    fi
    sleep 60
done
