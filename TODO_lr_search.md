# LR Schedule Optimization Search Plan

> 목표: 학습 epoch 200→100으로 줄이면서 동등 이상의 PESQ 달성
> 작성일: 2026-03-24

## Baseline Reference (ExponentialLR γ=0.99, β₁=0.8, 200ep)

| Fold | Best PESQ | Best Epoch | @100ep |
|------|-----------|------------|--------|
| fold1 | 2.6716 | 76 | 2.6716 |
| fold2 | 2.8354 | 137 | 2.7801 |
| fold3 | 2.8565 | 111 | 2.8281 |
| fold4 | 2.8974 | 134 | 2.8490 |
| fold5 | 2.7482 | 132 | 2.7121 |

## 변경사항 (Phase 0, 완료)

- `betas`: [0.8, 0.99] → **[0.9, 0.99]**
- `weight_decay`: 명시적 설정 (기존 AdamW 기본값 0.01)
- `scheduler`: ExponentialLR → **Cosine + Linear Warmup** (config에서 선택 가능)
- `conf/config.yaml`, `src/train.py` 수정 완료

---

## Phase 1 — Scheduler × LR 스크리닝 (fold4, 100ep × 4)

공통: `scheduler.type=cosine  betas=[0.9,0.99]  epochs=100  eval_every=100  cv.enabled=true  cv.fold_index=4`

| # | Name | Hydra Overrides | Status |
|---|------|-----------------|--------|
| 1 | `p1a_cos_lr5e4_wu10` | `lr=5e-4 scheduler.warmup_epochs=10 weight_decay=0.01` | [ ] |
| 2 | `p1b_cos_lr8e4_wu10` | `lr=8e-4 scheduler.warmup_epochs=10 weight_decay=0.01` | [ ] |
| 3 | `p1c_cos_lr8e4_wu10_wd001` | `lr=8e-4 scheduler.warmup_epochs=10 weight_decay=0.001` | [ ] |
| 4 | `p1d_cos_lr1e3_wu15` | `lr=1e-3 scheduler.warmup_epochs=15 weight_decay=0.01` | [ ] |

**실행 예시**:
```bash
./run_train.sh --fold 4 p1b_cos_lr8e4_wu10 \
    lr=8e-4 scheduler.warmup_epochs=10 weight_decay=0.01 epochs=100 eval_every=100
```

**판정 기준**:
- 성공: running_best@100ep ≥ 2.8974 (baseline fold4 best@134ep)
- B vs C 비교로 WD 민감도 판단 → **차이 < 0.01이면 Phase 2 스킵**

### Phase 1 결과 기록

| # | Config | best PESQ@100ep | best epoch | 비고 |
|---|--------|-----------------|------------|------|
| 1 | A | | | |
| 2 | B | | | |
| 3 | C | | | |
| 4 | D | | | |

**Phase 1 승자**: _______________
**B vs C 차이**: ___ → Phase 2: 진행 / 스킵

---

## Phase 2 — WD 미세조정 (조건부, fold4, 100ep × 0~2)

> Phase 1에서 B vs C PESQ 차이 ≥ 0.01일 때만 진행

Phase 1 승자의 config를 기반으로 weight_decay만 변경:

| # | Name | Hydra Overrides | Status |
|---|------|-----------------|--------|
| 5 | `p2e_wd005` | Phase 1 승자 + `weight_decay=0.005` | [ ] |
| 6 | `p2f_wd02` | Phase 1 승자 + `weight_decay=0.02` | [ ] |

### Phase 2 결과 기록

| # | Config | best PESQ@100ep | best epoch | 비고 |
|---|--------|-----------------|------------|------|
| 5 | E | | | |
| 6 | F | | | |

**최종 config 확정**: _______________

---

## Phase 3 — Cross-fold 검증 (fold2 + fold5, 100ep × 2)

최종 config로 baseline에서 수렴이 느렸던 fold에서 검증:

| # | Name | Fold | Baseline best | Status |
|---|------|------|---------------|--------|
| 7 | `p3_fold2_final` | fold2 | 2.8354 @137ep | [ ] |
| 8 | `p3_fold5_final` | fold5 | 2.7482 @132ep | [ ] |

**판정**: 두 fold 모두 best@100ep ≥ baseline best@full이면 Phase 4 진행

### Phase 3 결과 기록

| # | Fold | best PESQ@100ep | vs Baseline | Pass? |
|---|------|-----------------|-------------|-------|
| 7 | fold2 | | | |
| 8 | fold5 | | | |

---

## Phase 4 — 최종 5-fold CV (100ep × 5)

| # | Name | Fold | Status |
|---|------|------|--------|
| 9 | `final_fold1` | fold1 | [ ] |
| 10 | `final_fold2` | fold2 | [ ] |
| 11 | `final_fold3` | fold3 | [ ] |
| 12 | `final_fold4` | fold4 | [ ] |
| 13 | `final_fold5` | fold5 | [ ] |

**최종 config (Phase 1~3에서 확정 후 기입)**:
```bash
lr=___  scheduler.warmup_epochs=___  weight_decay=___  epochs=100
scheduler.type=cosine  betas=[0.9,0.99]
```

### Phase 4 결과 기록

| Fold | New best PESQ | Baseline best | Δ | New best epoch |
|------|---------------|---------------|---|----------------|
| fold1 | | 2.6716 | | |
| fold2 | | 2.8354 | | |
| fold3 | | 2.8565 | | |
| fold4 | | 2.8974 | | |
| fold5 | | 2.7482 | | |
| **Mean** | | **2.8018** | | |

---

## 총 실험 횟수

| Phase | 실험 수 | GPU 시간 (직렬) | GPU 시간 (병렬) |
|-------|---------|-----------------|-----------------|
| 1. 스크리닝 | 4 | 72h | 18h |
| 2. WD 조정 (조건부) | 0~2 | 0~36h | 0~18h |
| 3. Cross-fold | 2 | 36h | 18h |
| 4. 최종 CV | 5 | 90h | 18h |
| **합계** | **11~13** | **198~234h** | **54~72h** |
