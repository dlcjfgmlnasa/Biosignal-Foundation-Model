from __future__ import annotations

"""평가 패키지 — 메트릭, 시각화, downstream 평가 모듈."""

# 회귀/복원 메트릭
from eval._metrics import (
    compute_bland_altman,
    compute_mae,
    compute_mse,
    compute_pearson_r,
    pearson_r,
    regression_metrics,
)

# 분류 메트릭
from eval._metrics import (  # noqa: E402
    compute_auprc,
    compute_auroc,
    compute_f1,
    compute_sensitivity_specificity,
)

# 시각화
from eval._viz import plot_bland_altman, plot_reconstruction, plot_roc_curve

# 평가 모듈
from eval.fewshot import PrototypicalClassifier, compute_classification_metrics
from eval.forecasting import evaluate_forecasting
from eval.imputation import evaluate_imputation

__all__ = [
    # 회귀
    "compute_bland_altman",
    "compute_mae",
    "compute_mse",
    "compute_pearson_r",
    "pearson_r",
    "regression_metrics",
    # 분류
    "compute_auprc",
    "compute_auroc",
    "compute_f1",
    "compute_sensitivity_specificity",
    # 시각화
    "plot_bland_altman",
    "plot_reconstruction",
    "plot_roc_curve",
    # 평가 모듈
    "PrototypicalClassifier",
    "compute_classification_metrics",
    "evaluate_forecasting",
    "evaluate_imputation",
]
