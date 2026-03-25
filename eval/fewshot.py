"""Prototypical Network 기반 few-shot 분류 평가.

nn.Linear 등 학습 가능한 classification head 사용 금지 — 인코더 동결 후
latent space 거리(코사인 유사도) 기반 분류만 허용.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from data.collate import PackedBatch


# ── PrototypicalClassifier ────────────────────────────────────────


class PrototypicalClassifier:
    """비-파라미터 few-shot 분류기.

    학습 가능 파라미터 없음 (nn.Module 아님).
    인코더를 동결하고 ``extract_features()`` 출력으로만 동작.

    Usage::

        clf = PrototypicalClassifier()
        clf.fit(model, support_batches, support_labels)
        preds = clf.predict(model, query_batch)
        metrics = clf.evaluate(model, query_batches, query_labels)
    """

    def __init__(self) -> None:
        self.prototypes: torch.Tensor | None = None  # (num_classes, d_model)
        self.class_labels: list[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        model: torch.nn.Module,
        support_batches: list[PackedBatch],
        labels: list[int],
    ) -> None:
        """Support set으로부터 클래스별 프로토타입 벡터 계산.

        Parameters
        ----------
        model:
            ``extract_features()`` 를 가진 사전학습 모델.
        support_batches:
            단일(또는 소수) 샘플을 담은 ``PackedBatch`` 리스트.
        labels:
            ``support_batches`` 와 1:1 매핑되는 정수 클래스 레이블.
        """
        assert len(support_batches) == len(labels)

        class_embeds: dict[int, list[torch.Tensor]] = defaultdict(list)
        for batch, label in zip(support_batches, labels):
            for emb in _embed(model, batch):
                class_embeds[label].append(emb)

        self.class_labels = sorted(class_embeds.keys())
        self.prototypes = torch.stack(
            [torch.stack(class_embeds[c]).mean(dim=0) for c in self.class_labels],
        )  # (num_classes, d_model)

    def predict(
        self,
        model: torch.nn.Module,
        query_batch: PackedBatch,
    ) -> list[int]:
        """Cosine similarity 기반 클래스 할당.

        Returns
        -------
        list[int]
            각 샘플의 예측 클래스 레이블.
        """
        assert self.prototypes is not None, "fit()을 먼저 호출하세요."

        embeddings = _embed(model, query_batch)  # (n_queries, D)
        if embeddings.shape[0] == 0:
            return []

        prototypes = self.prototypes.to(embeddings.device)
        sim = F.cosine_similarity(
            embeddings.unsqueeze(1),   # (n_queries, 1, D)
            prototypes.unsqueeze(0),   # (1, num_classes, D)
            dim=-1,
        )  # (n_queries, num_classes)

        indices = sim.argmax(dim=-1)
        return [self.class_labels[i.item()] for i in indices]

    def predict_batches(
        self,
        model: torch.nn.Module,
        query_batches: list[PackedBatch],
    ) -> list[int]:
        """여러 배치에 대해 순차적으로 predict 수행."""
        preds: list[int] = []
        for batch in query_batches:
            preds.extend(self.predict(model, batch))
        return preds

    def evaluate(
        self,
        model: torch.nn.Module,
        query_batches: list[PackedBatch],
        labels: list[int],
    ) -> dict[str, float | list]:
        """Few-shot 분류 평가 — 메트릭 dict 반환.

        Returns
        -------
        dict with keys:
            ``accuracy``, ``balanced_accuracy``, ``macro_f1``,
            ``cohens_kappa``, ``confusion_matrix``.
        """
        preds = self.predict_batches(model, query_batches)
        assert len(preds) == len(labels)
        return compute_classification_metrics(preds, labels, self.class_labels)


# ── Feature extraction ────────────────────────────────────────────


@torch.no_grad()
def _embed(model: torch.nn.Module, batch: PackedBatch) -> torch.Tensor:
    """Batch → per-sample mean-pooled feature ``(n_samples, d_model)``.

    ``extract_features()`` 호출 후 ``patch_sample_id`` 로 샘플 분리,
    유효 패치(``patch_mask``)만 mean pooling.
    """
    model.eval()
    out = model.extract_features(batch)

    encoded = out["encoded"]                     # (B, N, D)
    patch_mask = out["patch_mask"]               # (B, N)
    patch_sample_id = out["patch_sample_id"]     # (B, N)

    B, N, D = encoded.shape
    embeddings: list[torch.Tensor] = []

    for b in range(B):
        valid = patch_mask[b]                    # (N,)
        sid = patch_sample_id[b]                 # (N,)
        enc = encoded[b]                         # (N, D)

        for uid in sid[valid].unique():
            if uid == 0:
                continue
            mask = valid & (sid == uid)
            embeddings.append(enc[mask].mean(dim=0))  # (D,)

    if not embeddings:
        return torch.empty(0, D, device=encoded.device)

    return torch.stack(embeddings)  # (n_samples, D)


# ── Classification metrics ────────────────────────────────────────


def compute_classification_metrics(
    y_pred: list[int],
    y_true: list[int],
    class_labels: list[int] | None = None,
) -> dict[str, float | list]:
    """분류 메트릭 (sklearn 미사용).

    Returns
    -------
    dict with keys:
        ``accuracy``, ``balanced_accuracy``, ``macro_f1``,
        ``cohens_kappa``, ``confusion_matrix``.
    """
    if class_labels is None:
        class_labels = sorted(set(y_true) | set(y_pred))

    n = len(y_true)
    k = len(class_labels)
    idx = {label: i for i, label in enumerate(class_labels)}

    # --- Confusion matrix ---
    cm = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        ti, pi = idx.get(t, -1), idx.get(p, -1)
        if ti >= 0 and pi >= 0:
            cm[ti][pi] += 1

    # --- Accuracy ---
    correct = sum(cm[i][i] for i in range(k))
    accuracy = correct / n if n > 0 else 0.0

    # --- Per-class precision / recall / F1 ---
    recalls, f1s = [], []
    for i in range(k):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(row[i] for row in cm) - tp

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0

        recalls.append(rec)
        f1s.append(f1)

    balanced_accuracy = sum(recalls) / k if k > 0 else 0.0
    macro_f1 = sum(f1s) / k if k > 0 else 0.0

    # --- Cohen's Kappa ---
    pe = sum(
        sum(cm[i]) * sum(row[i] for row in cm) for i in range(k)
    ) / (n * n) if n > 0 else 0.0
    kappa = (accuracy - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "cohens_kappa": kappa,
        "confusion_matrix": cm,
    }
