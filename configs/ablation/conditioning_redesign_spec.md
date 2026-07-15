# Conditioning Redesign Spec — A(게이팅) + B(peak enrichment)

> Status: **CONFIRMED (2026-07-15)**. A·B는 다음 from-scratch 재사전학습에 baking. C(median/IQR)는 드롭.
> ablation 격리 단계는 시간 제약으로 생략 — "명확한 A는 신뢰, B는 저비용 베팅"으로 한 방에 감. 안전장치(§5)로 헷지.
> 역할분담: 이 스펙 = Claude 작성. 구현·재학습 = 사용자(서버).

## 0. 배경 / 근거 (실측)

문제: `model/biosignal_model.py` `_encode`에서 loc/scale이 **signal_type 무관 공용 `cond_proj`**(detach 없음)를 거쳐 전 레이어 `LSCNorm` AdaLN에 주입됨. unitless 모달리티(**PPG=2, resp_imp=7**)는 이 loc/scale이 device-arbitrary(장비 게인/baseline) → device confound / shortcut.

- **PI 실측**: PPG 파형 AC/DC vs 모니터 reported PI(`PLETH_PERF_REL`) pooled Spearman **0.064**, 크기 20–55× 어긋남 → 파형은 오토스케일이라 관류정보 소실 → **게이팅이 임상정보 손실 0** 확인.
- **잔여 누출**: z-score(크기 제거) 후에도 PPG 형태만으로 SNUADC vs Intellivue **AUROC 0.72–0.78** (HR-only 0.542) → 게이팅은 필요·주처방이나 형태 잔여는 content 경로라 못 잡음 → device-null eval 필수(§6).
- **peak 가치**: 비대칭 생체파형은 max=SBP/ETCO2/PIP, min=DBP/PEEP가 mean/std로 복원 불가. 특히 **CO2는 mean이 나쁜 요약, max=ETCO2가 THE 임상값**.
- **비용**: scatter_max/min **2.2ms** vs 분위수(정렬) **284ms** (~130×). CO2/AWP는 true max≈p99.

## 1. Change A — per-modality 게이팅 (loc/scale conditioning)

**무엇**: signal_type ∈ {PPG=2, resp_imp=7} 토큰의 `ada_cond`(loc/scale conditioning)를 0으로.

**어디**: `model/biosignal_model.py` `_encode`, `ada_cond = self.cond_proj(loc_scale)`(≈L431) **직후**, `ada_cond = ada_cond * valid_token` 전. ⚠️ 마스크는 반드시 **cond_proj 출력**에 — 입력 `loc_scale`에 걸면 cond_proj bias 때문에 의미 깨짐(architect).

```python
ada_cond = self.cond_proj(loc_scale)                 # (B, N, d_cond)
if self.gate_unitless_cond:                          # toggle flag
    gated = (patch_signal_types == 2) | (patch_signal_types == 7)   # PPG, resp_imp (B,N)
    ada_cond = ada_cond * (~gated).unsqueeze(-1)
ada_cond = ada_cond * valid_token
```

- `patch_signal_types`는 이미 L385에서 계산됨 → **scaler plumbing 불필요**.
- **효과**: 게이팅된 토큰의 LSCNorm은 `norm(x)·(1+bias_γ)+bias_β`(학습된 static affine)로 fallback — 제대로 된 학습 norm, loc/scale/device 의존만 제거.
- **유지**: modality 임베딩(sig_emb)·z-score 정규화 그대로. loc/scale **conditioning만** 차단.
- **체크포인트**: 호환(입력 마스크, 차원 불변) → warm-start 가능.
- **플래그**: `gate_unitless_cond: bool = True`.

## 2. Change B — conditioning enrichment [mean,std] → [mean,std,max,min]

**무엇**: per-variate 통계를 (loc, scale) → (loc, scale, max, min)으로 확장. calibrated 전용(PPG/resp-imp는 A가 4개 다 게이팅).

**스케일러** `module/packed_scaler.py` `PackedStdScaler._get_loc_scale` — 전 그룹 균일 계산(signal_type 안 받음):

```python
# winsorize 가드: 스파이크가 max/min을 날리지 않게 clamp 후 scatter (robust @ scatter 비용)
k = 5.0
capped = target.clamp(min=loc - k*scale, max=loc + k*scale)   # loc/scale은 위에서 계산됨
group_max = torch.full((B, n_groups, D), float("-inf"), dtype=target.dtype, device=target.device)
group_max.scatter_reduce_(1, gk, capped, reduce="amax", include_self=False)
group_min = torch.full((B, n_groups, D), float("inf"),  dtype=target.dtype, device=target.device)
group_min.scatter_reduce_(1, gk, capped, reduce="amin", include_self=False)
max_ = group_max.gather(1, gk); min_ = group_min.gather(1, gk)
# padding 초기화 (loc/scale과 동일 패턴)
max_ = max_.masked_fill(padding, 0.0); min_ = min_.masked_fill(padding, 0.0)
return loc, scale, max_, min_
```

⚠️ 함정(data-engineer): `include_self=False` + `±inf` 초기화 필수(ECG는 zero-mean·음수 가능, init 0이면 오clamp). 빈 그룹의 ±inf는 gather 안 되니 안전. observed_mask는 현재 all-ones라 무해하나, 향후 NaN 도입 시 `capped.masked_fill(~obs, ±inf)` 필요(주석 박기).

**모델** `_encode`: patch_starts에서 patch_max/patch_min 샘플 → `cat([patch_loc, patch_scale, patch_max, patch_min])` (B,N,4) → `cond_proj = Sequential(Linear(4, d_cond), SiLU, Linear(d_cond, d_cond))`.

- **RAW 물리 단위 유지** — max/min을 z-정규화 금지(그래야 max=SBP·ETCO2 의미 살아있음).
- **체크포인트**: **비호환**(cond_proj 2→4) → from-scratch. 어차피 신규 코퍼스 재학습이라 **추가비용 0**.
- **플래그**: `enrich_cond_peak: bool = True` (생성 시 cond_proj 입력 2 vs 4 선택).

## 3. Change C — DROPPED

전역 median/IQR robust scaler 드롭. mean/std 유지. (근거: scatter로 median/IQR 불가(O(L log L) 정렬)·파서가 이미 신호별 아티팩트 제거·loc를 median으로 바꾸면 AdaLN "레벨" 의미 변질·A의 대체재 아님. robustness 필요 시 파스단 outlier 게이트 강화가 더 쌈.)

## 4. 시퀀싱 (no-ablation, one-shot)

A+B를 신규 확장 코퍼스 **단일 from-scratch 재사전학습**에 함께 baking. ablation 격리 생략.

## 5. 안전장치 (near-zero cost, "믿고 가되 눈 감진 말자")

1. **A·B 토글 플래그**(§1·§2) — 최종 모델이 baseline 미달 시 재파싱 없이 끄고 원인 좁힘.
2. **현 30M ckpt = 이겨야 할 baseline** — eval suite에서 못 이기면 신호.
3. **최종 모델 device-null eval 1회**(§6) — 게이팅이 누출 줄였는지 + 논문 device 통제 정직 기재.

## 6. Device-null eval 프로토콜 (post-hoc, 학습된 모델 1개)

- **부트스트랩은 신규 불필요** — `scripts/aggregate_ablation_results.py::paired_bootstrap_delta`(L177) 재사용. 두 예측셋을 같은 환자 resample에서 Δ + Δ-CI + two-sided p 계산. full vs neutralized는 같은 모델·데이터·환자라 완벽 정합.
- **신규 필요 = neutralization 훅만**: `extract_features`/`forward`에 PPG(·resp-imp) loc/scale을 canonical(loc=모집단 중앙값, scale=1)로 override하는 인자 추가 → full·neutralized 예측 2벌 dump(같은 patient_id).
- **대상**: unitless 노출 task(Arrhythmia PPG-only, Hypoxemia 등). 소스 혼합/cross-device 검증에서 특히.
- **판정**: `paired_bootstrap_delta(full_oof, neutral_oof, auroc)` Δ-CI가 0 포함 → device 몫 없음(게이팅 충분). 0 배제 → 잔여 device 몫 정량화 → 해당 task는 neutralized feature로 headline 보고.
- (보강) covariate-only null: device/source를 feature로 한 probe AUROC와 실모델 비교(marginal CI 겹치면 shortcut 의심).

## 7. 손댈 파일

- `model/biosignal_model.py`: `__init__`(cond_proj 2→4, 플래그 2개), `_encode`(A 게이트 ≈L431, B의 cat 4 + patch_max/min 샘플).
- `module/packed_scaler.py`: `PackedStdScaler._get_loc_scale`(winsorize + scatter max/min, 4-tuple 반환). ⚠️ 반환 시그니처 변경 → `_encode`의 언패킹·`PackedNOPScaler`/`PackedAbsMeanScaler`도 4-tuple로 맞추기(또는 4번째·5번째 optional).
- `configs/ablation/variants.yaml`: (선택) 사후 격리용 `07a_ppg_resp_gate`(A, ckpt 호환)·`07b_enrich_maxmin`(B, from-scratch) 등록.
- `downstream/`: device-null neutralization 훅(신규) + `paired_bootstrap_delta` 재사용 글루.

## 8. signal_type 레퍼런스 (v2)

게이팅: **PPG=2, resp_imp=7**. calibrated([mean,std,max,min] 전부): ECG=0(bandpass→loc≈0, 무해)·ABP=1·CVP=3·CO2=4·AWP=5·ICP=6·resp_flow=8.
