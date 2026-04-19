# -*- coding: utf-8 -*-
"""
NBA Team-Season Win Rate Prediction Pipeline (Leakage-Free, Year-Based Split)
=============================================================================
Stage 1  : Preprocessing & collinearity filter  (fit on train years only)
Stage 2a : LassoCV feature selection             (fit on train years only)
Stage 2b : ElasticNetCV feature selection         (fit on train years only)
Stage 3  : XGBoost + GridSearchCV + SHAP

Split modes
-----------
- "random"       : row-level random train_test_split (legacy)
- "year_holdout" : train on earlier seasons, test on later seasons
- "rolling_year" : expanding-window backtest, one test year at a time

All preprocessing rules (imputation, scaler, variance filter, collinearity
filter) are fit exclusively on training data for every split.

Usage:
    from nba_win_rate_pipeline import run_full_pipeline, load_metrics_from_csvs

    # Point at your licensed CSVs under data/processed/, or sample_data/processed/
    df = load_metrics_from_csvs("data/processed")
    # Mode A: fixed year holdout
    results = run_full_pipeline(df, split_mode="year_holdout",
                                train_years=list(range(2014,2023)),
                                test_years=[2023,2024,2025])
    # Mode B: rolling backtest
    results = run_full_pipeline(df, split_mode="rolling_year",
                                min_train_years=5)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
})

# Consistent colour palette for the three models
_CLR_LASSO = "#1565C0"
_CLR_ENET = "#2E7D32"
_CLR_XGB = "#E65100"


# ===================================================================
# Utility helpers
# ===================================================================
def resolve_target_column(df: pd.DataFrame, target_col: str) -> str:
    if target_col in df.columns:
        return target_col
    alt = "Win Rate" if target_col.lower() == "win_rate" else target_col
    if alt in df.columns:
        return alt
    raise KeyError(
        f"Target column not found: {target_col}; "
        f"available: {list(df.columns)[:20]}..."
    )


def infer_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    id_columns: Optional[List[str]] = None,
) -> List[str]:
    id_columns = id_columns or []
    resolved_target = resolve_target_column(df, target_col)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {resolved_target, *id_columns}
    for c in ("Unnamed: 0", "Season", "season", "year", "Year"):
        if c in numeric:
            exclude.add(c)
    return [c for c in numeric if c not in exclude]


# ===================================================================
# Train / Test split strategies
# ===================================================================
def _random_split_indices(
    n_samples: int,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    tr, te = train_test_split(
        np.arange(n_samples), test_size=test_size, random_state=random_state,
    )
    return np.array(tr), np.array(te)


def get_year_based_split(
    df: pd.DataFrame,
    train_years: List[int],
    test_years: List[int],
    year_col: str = "year",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return row indices for a year-based train/test split.

    Guarantees:
      - No year appears in both train and test.
      - Both splits are non-empty.
    """
    if year_col not in df.columns:
        raise KeyError(
            f"Year column '{year_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)[:20]}"
        )

    overlap = set(train_years) & set(test_years)
    if overlap:
        raise ValueError(
            f"train_years and test_years overlap on: {sorted(overlap)}"
        )

    train_mask = df[year_col].isin(train_years)
    test_mask = df[year_col].isin(test_years)

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    if len(train_idx) == 0:
        raise ValueError(
            f"No training samples found for years {train_years}. "
            f"Available years: {sorted(df[year_col].unique())}"
        )
    if len(test_idx) == 0:
        raise ValueError(
            f"No test samples found for years {test_years}. "
            f"Available years: {sorted(df[year_col].unique())}"
        )

    return train_idx, test_idx


def _build_rolling_folds(
    df: pd.DataFrame,
    year_col: str = "year",
    min_train_years: int = 3,
) -> List[Tuple[List[int], List[int]]]:
    all_years = sorted(df[year_col].unique())
    folds = []
    for i in range(min_train_years, len(all_years)):
        tr_years = all_years[:i]
        te_years = [all_years[i]]
        folds.append((list(tr_years), list(te_years)))
    return folds


# ===================================================================
# Stage 1: Preprocessing & collinearity filter (Leakage-Free)
# ===================================================================
@dataclass
class Stage1Artifacts:
    impute_values: Dict[str, float]
    keep_cols_after_variance: List[str]
    scaler: StandardScaler
    keep_cols_after_collinear: List[str]
    dropped_pairs_log: List[Tuple[str, str, str]] = field(default_factory=list)
    n_features_before: int = 0
    n_features_after: int = 0


@dataclass
class Stage1Result:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    artifacts: Stage1Artifacts
    n_features_raw: int = 0


def _pearson_corr_with_target(series: pd.Series, target: pd.Series) -> float:
    aligned = pd.concat([series, target], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def _remove_collinear_features_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    corr_threshold: float = 0.85,
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    X_work = X_train.copy()
    dropped_log: List[Tuple[str, str, str]] = []

    def target_abs_corrs() -> dict:
        return {
            col: abs(_pearson_corr_with_target(X_work[col], y_train))
            for col in X_work.columns
        }

    while True:
        cols = list(X_work.columns)
        if len(cols) < 2:
            break
        cmat = X_work.corr(method="pearson")
        pair_found = None
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                ci, cj = cols[i], cols[j]
                r = cmat.loc[ci, cj]
                if np.isnan(r):
                    continue
                if abs(r) > corr_threshold:
                    pair_found = (ci, cj)
                    break
            if pair_found:
                break
        if not pair_found:
            break

        ci, cj = pair_found
        ta = target_abs_corrs()
        ai, aj = ta[ci], ta[cj]
        if ai < aj:
            drop_col = ci
        elif aj < ai:
            drop_col = cj
        else:
            drop_col = sorted([ci, cj])[0]

        dropped_log.append((ci, cj, drop_col))
        X_work = X_work.drop(columns=[drop_col])

    return list(X_work.columns), dropped_log


def _fit_stage1(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    corr_threshold: float,
) -> Stage1Artifacts:
    impute_values = X_train_raw.mean().to_dict()
    impute_values = {k: (v if pd.notna(v) else 0.0) for k, v in impute_values.items()}
    X_filled = X_train_raw.fillna(impute_values)

    std = X_filled.std(numeric_only=True)
    keep_var = std[std > 1e-12].index.tolist()
    X_filled = X_filled[keep_var]
    n_before = len(keep_var)

    scaler = StandardScaler()
    scaler.fit(X_filled)
    X_scaled = pd.DataFrame(
        scaler.transform(X_filled), index=X_filled.index, columns=X_filled.columns,
    )
    X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    keep_cols, dropped_log = _remove_collinear_features_train(
        X_scaled, y_train, corr_threshold,
    )
    n_after = len(keep_cols)

    return Stage1Artifacts(
        impute_values=impute_values,
        keep_cols_after_variance=keep_var,
        scaler=scaler,
        keep_cols_after_collinear=keep_cols,
        dropped_pairs_log=dropped_log,
        n_features_before=n_before,
        n_features_after=n_after,
    )


def _transform_stage1(
    X_raw: pd.DataFrame,
    artifacts: Stage1Artifacts,
) -> pd.DataFrame:
    cols_needed = artifacts.keep_cols_after_variance
    X_raw_c = X_raw.copy()
    for c in cols_needed:
        if c not in X_raw_c.columns:
            X_raw_c[c] = 0.0
    X_sub = X_raw_c[cols_needed].copy()
    X_sub = X_sub.fillna(artifacts.impute_values).fillna(0.0)

    X_np = artifacts.scaler.transform(X_sub)
    X_scaled = pd.DataFrame(X_np, index=X_sub.index, columns=cols_needed)
    X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    final_cols = [c for c in artifacts.keep_cols_after_collinear if c in X_scaled.columns]
    return X_scaled[final_cols]


def stage1_preprocess(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_col: str = "win_rate",
    corr_threshold: float = 0.85,
    id_columns: Optional[List[str]] = None,
    verbose: bool = True,
) -> Stage1Result:
    target_col = resolve_target_column(df, target_col)
    feature_cols = infer_feature_columns(df, target_col, id_columns)
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    n_features_raw = len(feature_cols)
    y = df[target_col].astype(float)
    X_all_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    X_train_raw = X_all_raw.iloc[train_idx]
    X_test_raw = X_all_raw.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    artifacts = _fit_stage1(X_train_raw, y_train, corr_threshold)

    X_train_clean = _transform_stage1(X_train_raw, artifacts)
    X_test_clean = _transform_stage1(X_test_raw, artifacts)

    final_features = list(X_train_clean.columns)

    if verbose:
        print("=" * 60)
        print("[Stage 1] Preprocessing & collinearity filter")
        print("=" * 60)
        print(f"  Raw feature count         : {n_features_raw}")
        print(f"  After variance + scaling  : {artifacts.n_features_before}")
        print(f"  After collinearity filter : {artifacts.n_features_after}")
        print(f"  Features removed          : {artifacts.n_features_before - artifacts.n_features_after}")
        if artifacts.dropped_pairs_log:
            print(f"  ({len(artifacts.dropped_pairs_log)} collinearity removal decisions)")
        print(f"  Train samples: {len(X_train_clean)}  |  Test samples: {len(X_test_clean)}")
        print(f"  [Leakage-free] All rules fit on train only, test only transformed.")
        print()

    return Stage1Result(
        X_train=X_train_clean,
        X_test=X_test_clean,
        y_train=y_train,
        y_test=y_test,
        feature_names=final_features,
        artifacts=artifacts,
        n_features_raw=n_features_raw,
    )


# ===================================================================
# CV splitter helper
# ===================================================================
def _make_cv_splitter(
    n_train: int,
    use_time_series_cv: bool,
    random_state: int,
    n_splits: int = 5,
) -> Union[KFold, TimeSeriesSplit]:
    if use_time_series_cv:
        safe_splits = min(n_splits, max(2, n_train // 15))
        safe_splits = max(2, min(safe_splits, n_train - 1))
        return TimeSeriesSplit(n_splits=safe_splits)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# ===================================================================
# Stage 2a: LassoCV feature selection
# ===================================================================
@dataclass
class Stage2Result:
    lasso: LassoCV
    selected_features: List[str]
    best_alpha: float
    train_r2: float
    train_rmse: float
    test_r2: float
    test_rmse: float
    train_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    test_pred: np.ndarray = field(default_factory=lambda: np.array([]))


def stage2_lasso_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
    use_time_series_cv: bool = False,
    max_iter: int = 20000,
    verbose: bool = True,
) -> Stage2Result:
    cv_lasso = _make_cv_splitter(
        len(X_train), use_time_series_cv, random_state,
    )

    alphas = np.logspace(-4, 1, 50)
    lasso = LassoCV(
        alphas=alphas, cv=cv_lasso, random_state=random_state,
        max_iter=max_iter, n_jobs=-1,
    )
    lasso.fit(X_train, y_train)

    pred_train = lasso.predict(X_train)
    pred_test = lasso.predict(X_test)

    train_r2 = r2_score(y_train, pred_train)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, pred_train)))
    test_r2 = r2_score(y_test, pred_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))

    names = list(X_train.columns)
    selected = [names[i] for i in range(len(names)) if lasso.coef_[i] != 0]

    if verbose:
        print("=" * 60)
        print("[Stage 2a] LassoCV feature selection (L1)")
        print("=" * 60)
        print(f"  Best alpha         : {lasso.alpha_:.6g}")
        print(f"  Non-zero features  : {len(selected)}")
        print(f"  Train R^2: {train_r2:.4f}  |  RMSE: {train_rmse:.4f}")
        print(f"  Test  R^2: {test_r2:.4f}  |  RMSE: {test_rmse:.4f}")
        print(f"  [Leakage-free] LassoCV fit on train only.")
        print()

    return Stage2Result(
        lasso=lasso, selected_features=selected,
        best_alpha=float(lasso.alpha_),
        train_r2=train_r2, train_rmse=train_rmse,
        test_r2=test_r2, test_rmse=test_rmse,
        train_pred=np.array(pred_train),
        test_pred=np.array(pred_test),
    )


def plot_lasso_coefficients(
    lasso: LassoCV,
    feature_names: List[str],
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    coef = lasso.coef_
    nonzero = [(feature_names[i], coef[i]) for i in range(len(coef)) if coef[i] != 0]
    if not nonzero:
        print("[Warning] All Lasso coefficients are zero; skipping plot.")
        return

    nonzero.sort(key=lambda x: abs(x[1]), reverse=True)
    names = [x[0] for x in nonzero][::-1]
    values = [x[1] for x in nonzero][::-1]

    fig, ax = plt.subplots(figsize=figsize)
    colors = [_CLR_LASSO if v >= 0 else "coral" for v in values]
    ax.barh(names, values, color=colors, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lasso coefficient (standardized features)")
    ax.set_ylabel("Feature")
    ax.set_title("Non-zero Lasso coefficients (sorted by absolute value)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ===================================================================
# Stage 2b: ElasticNetCV feature selection
# ===================================================================
@dataclass
class Stage2ElasticNetResult:
    model: ElasticNetCV
    selected_features: List[str]
    best_alpha: float
    best_l1_ratio: float
    train_r2: float
    train_rmse: float
    test_r2: float
    test_rmse: float
    train_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    test_pred: np.ndarray = field(default_factory=lambda: np.array([]))


def stage2_elasticnet_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
    use_time_series_cv: bool = False,
    max_iter: int = 20000,
    verbose: bool = True,
) -> Stage2ElasticNetResult:
    """Elastic Net with CV-tuned alpha and l1_ratio. Fit on train only."""
    cv_enet = _make_cv_splitter(
        len(X_train), use_time_series_cv, random_state,
    )

    alphas = np.logspace(-4, 1, 50)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    enet = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=cv_enet,
        random_state=random_state,
        max_iter=max_iter,
        n_jobs=-1,
    )
    enet.fit(X_train, y_train)

    pred_train = enet.predict(X_train)
    pred_test = enet.predict(X_test)

    train_r2 = r2_score(y_train, pred_train)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, pred_train)))
    test_r2 = r2_score(y_test, pred_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))

    names = list(X_train.columns)
    selected = [names[i] for i in range(len(names)) if enet.coef_[i] != 0]

    if verbose:
        print("=" * 60)
        print("[Stage 2b] ElasticNetCV feature selection (L1 + L2)")
        print("=" * 60)
        print(f"  Best alpha         : {enet.alpha_:.6g}")
        print(f"  Best l1_ratio      : {enet.l1_ratio_:.4f}")
        print(f"  Non-zero features  : {len(selected)}")
        print(f"  Train R^2: {train_r2:.4f}  |  RMSE: {train_rmse:.4f}")
        print(f"  Test  R^2: {test_r2:.4f}  |  RMSE: {test_rmse:.4f}")
        print(f"  [Leakage-free] ElasticNetCV fit on train only.")
        print()

    return Stage2ElasticNetResult(
        model=enet,
        selected_features=selected,
        best_alpha=float(enet.alpha_),
        best_l1_ratio=float(enet.l1_ratio_),
        train_r2=train_r2, train_rmse=train_rmse,
        test_r2=test_r2, test_rmse=test_rmse,
        train_pred=np.array(pred_train),
        test_pred=np.array(pred_test),
    )


def plot_elasticnet_coefficients(
    model: ElasticNetCV,
    feature_names: List[str],
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    coef = model.coef_
    nonzero = [(feature_names[i], coef[i]) for i in range(len(coef)) if coef[i] != 0]
    if not nonzero:
        print("[Warning] All Elastic Net coefficients are zero; skipping plot.")
        return

    nonzero.sort(key=lambda x: abs(x[1]), reverse=True)
    names = [x[0] for x in nonzero][::-1]
    values = [x[1] for x in nonzero][::-1]

    fig, ax = plt.subplots(figsize=figsize)
    colors = [_CLR_ENET if v >= 0 else "coral" for v in values]
    ax.barh(names, values, color=colors, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Elastic Net coefficient (standardized features)")
    ax.set_ylabel("Feature")
    ax.set_title("Non-zero Elastic Net coefficients (sorted by absolute value)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ===================================================================
# Stage 3: XGBoost + GridSearchCV + SHAP
# ===================================================================
@dataclass
class Stage3Result:
    model: XGBRegressor
    best_params: dict
    train_r2: float
    train_rmse: float
    test_r2: float
    test_rmse: float
    shap_values: Optional[np.ndarray] = None
    train_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    test_pred: np.ndarray = field(default_factory=lambda: np.array([]))


def stage3_xgboost_shap(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_list: List[str],
    random_state: int = 42,
    use_time_series_cv: bool = False,
    grid_max_depth: Optional[List[int]] = None,
    grid_learning_rate: Optional[List[float]] = None,
    show_plot: bool = True,
    shap_save_path: Optional[str] = None,
    verbose: bool = True,
) -> Stage3Result:
    grid_max_depth = grid_max_depth or [3, 4, 5]
    grid_learning_rate = grid_learning_rate or [0.01, 0.05, 0.1]

    X_tr = X_train[feature_list].copy()
    X_te = X_test[feature_list].copy()

    cv_grid = _make_cv_splitter(len(X_tr), use_time_series_cv, random_state)

    base = XGBRegressor(
        objective="reg:squarederror", random_state=random_state,
        n_estimators=300, reg_alpha=0.1, reg_lambda=1.0,
        subsample=0.85, colsample_bytree=0.85,
    )

    param_grid = {
        "max_depth": grid_max_depth,
        "learning_rate": grid_learning_rate,
        "min_child_weight": [1, 3],
    }

    grid = GridSearchCV(
        base, param_grid, cv=cv_grid,
        scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=0,
    )
    grid.fit(X_tr, y_train)

    best: XGBRegressor = grid.best_estimator_

    pred_tr = best.predict(X_tr)
    pred_te = best.predict(X_te)

    train_r2 = r2_score(y_train, pred_tr)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, pred_tr)))
    test_r2 = r2_score(y_test, pred_te)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, pred_te)))

    if verbose:
        print("=" * 60)
        print("[Stage 3] XGBoost + GridSearchCV + SHAP")
        print("=" * 60)
        print(f"  Best params   : {grid.best_params_}")
        print(f"  Train R^2: {train_r2:.4f}  |  RMSE: {train_rmse:.4f}")
        print(f"  Test  R^2: {test_r2:.4f}  |  RMSE: {test_rmse:.4f}")
        print(f"  [Leakage-free] GridSearchCV fit on train only; SHAP on test set.")
        print()

    explainer = shap.TreeExplainer(best)
    sv = explainer.shap_values(X_te)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        sv, X_te, feature_names=feature_list, show=False, plot_size=(12, 8),
    )
    plt.title(
        "SHAP summary: impact of each feature on predicted win rate "
        "(color = feature value)", fontsize=12,
    )
    plt.tight_layout()
    if shap_save_path:
        plt.savefig(shap_save_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close("all")

    return Stage3Result(
        model=best, best_params=dict(grid.best_params_),
        train_r2=train_r2, train_rmse=train_rmse,
        test_r2=test_r2, test_rmse=test_rmse,
        shap_values=np.array(sv),
        train_pred=np.array(pred_tr),
        test_pred=np.array(pred_te),
    )


# ===================================================================
# Internal: run a single fold (Stage 1 -> 2a -> 2b -> 3)
# ===================================================================
def _run_single_fold(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_col: str,
    corr_threshold: float,
    id_columns: Optional[List[str]],
    random_state: int,
    use_time_series_cv: bool,
    show_plots: bool,
    save_extra_plots: bool,
    out_dir: Path,
    verbose: bool = True,
) -> dict:
    """Run the full pipeline (Stage 1 -> 2a Lasso -> 2b ENet -> 3 XGB)."""
    s1 = stage1_preprocess(
        df, train_idx=train_idx, test_idx=test_idx,
        target_col=target_col, corr_threshold=corr_threshold,
        id_columns=id_columns, verbose=verbose,
    )

    # Stage 2a: LassoCV
    s2 = stage2_lasso_selection(
        X_train=s1.X_train, X_test=s1.X_test,
        y_train=s1.y_train, y_test=s1.y_test,
        random_state=random_state,
        use_time_series_cv=use_time_series_cv,
        verbose=verbose,
    )

    if not s2.selected_features:
        raise RuntimeError(
            "Lasso selected zero features. Try a wider alpha range or "
            "lower corr_threshold."
        )

    lasso_coef_path = str(out_dir / "lasso_coefficients.png") if save_extra_plots else None
    plot_lasso_coefficients(
        s2.lasso, s1.feature_names,
        save_path=lasso_coef_path, show=show_plots,
    )

    # Stage 2b: ElasticNetCV
    s2_enet = stage2_elasticnet_selection(
        X_train=s1.X_train, X_test=s1.X_test,
        y_train=s1.y_train, y_test=s1.y_test,
        random_state=random_state,
        use_time_series_cv=use_time_series_cv,
        verbose=verbose,
    )

    enet_coef_path = str(out_dir / "elasticnet_coefficients.png") if save_extra_plots else None
    plot_elasticnet_coefficients(
        s2_enet.model, s1.feature_names,
        save_path=enet_coef_path, show=show_plots,
    )

    # Stage 3: XGBoost (uses Lasso-selected features)
    shap_path = str(out_dir / "shap_summary.png") if save_extra_plots else None
    s3 = stage3_xgboost_shap(
        X_train=s1.X_train, X_test=s1.X_test,
        y_train=s1.y_train, y_test=s1.y_test,
        feature_list=s2.selected_features,
        random_state=random_state,
        use_time_series_cv=use_time_series_cv,
        show_plot=show_plots, shap_save_path=shap_path,
        verbose=verbose,
    )

    return {"stage1": s1, "stage2": s2, "stage2_enet": s2_enet, "stage3": s3}


# ===================================================================
# Rolling year backtest
# ===================================================================
def run_rolling_year_backtest(
    df: pd.DataFrame,
    target_col: str = "win_rate",
    corr_threshold: float = 0.85,
    id_columns: Optional[List[str]] = None,
    year_col: str = "year",
    min_train_years: int = 3,
    random_state: int = 42,
    use_time_series_cv: bool = True,
    show_plots: bool = False,
    save_summary_plot: bool = True,
    plot_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Expanding-window backtest across seasons.

    For each fold the full leakage-free pipeline (including Elastic Net)
    is re-run from scratch: preprocessing is fit on training years only.
    """
    target_col = resolve_target_column(df, target_col)
    df = df.copy()
    df = df.dropna(subset=[target_col])
    df = df.reset_index(drop=True)

    out_dir = Path(plot_dir) if plot_dir else Path(".")
    folds = _build_rolling_folds(df, year_col, min_train_years)

    if not folds:
        raise ValueError(
            f"Not enough years for rolling backtest with min_train_years={min_train_years}. "
            f"Available years: {sorted(df[year_col].unique())}"
        )

    print("=" * 60)
    print(f"[Rolling year backtest] {len(folds)} folds, "
          f"min_train_years={min_train_years}")
    print("=" * 60)
    print()

    records = []

    for fold_i, (tr_years, te_years) in enumerate(folds):
        test_year = te_years[0]
        train_idx, test_idx = get_year_based_split(df, tr_years, te_years, year_col)

        print(f"--- Fold {fold_i+1}: train {tr_years[0]}-{tr_years[-1]} "
              f"({len(train_idx)} rows)  |  test {test_year} "
              f"({len(test_idx)} rows) ---")

        try:
            res = _run_single_fold(
                df, train_idx, test_idx,
                target_col=target_col,
                corr_threshold=corr_threshold,
                id_columns=id_columns,
                random_state=random_state,
                use_time_series_cv=use_time_series_cv,
                show_plots=False,
                save_extra_plots=False,
                out_dir=out_dir,
                verbose=False,
            )

            s2 = res["stage2"]
            s2e = res["stage2_enet"]
            s3 = res["stage3"]

            records.append({
                "test_year": test_year,
                "train_years": f"{tr_years[0]}-{tr_years[-1]}",
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "selected_feature_count": len(s2.selected_features),
                "lasso_best_alpha": s2.best_alpha,
                "lasso_train_r2": s2.train_r2,
                "lasso_test_r2": s2.test_r2,
                "lasso_train_rmse": s2.train_rmse,
                "lasso_test_rmse": s2.test_rmse,
                "enet_best_alpha": s2e.best_alpha,
                "enet_best_l1_ratio": s2e.best_l1_ratio,
                "enet_selected_feature_count": len(s2e.selected_features),
                "enet_train_r2": s2e.train_r2,
                "enet_test_r2": s2e.test_r2,
                "enet_train_rmse": s2e.train_rmse,
                "enet_test_rmse": s2e.test_rmse,
                "xgb_train_r2": s3.train_r2,
                "xgb_test_r2": s3.test_r2,
                "xgb_train_rmse": s3.train_rmse,
                "xgb_test_rmse": s3.test_rmse,
                "xgb_best_params": str(s3.best_params),
                "selected_features": ", ".join(s2.selected_features),
            })

            print(f"  Lasso      test R^2={s2.test_r2:.4f}  RMSE={s2.test_rmse:.4f}")
            print(f"  ElasticNet test R^2={s2e.test_r2:.4f}  RMSE={s2e.test_rmse:.4f}")
            print(f"  XGBoost    test R^2={s3.test_r2:.4f}  RMSE={s3.test_rmse:.4f}")
            print()

        except Exception as e:
            print(f"  [SKIP] Fold failed: {e}")
            print()
            continue

    result_df = pd.DataFrame(records)

    if len(result_df) > 0:
        print("=" * 60)
        print("[Rolling backtest summary]")
        print("=" * 60)
        print(f"  Avg Lasso      test R^2 : {result_df['lasso_test_r2'].mean():.4f}")
        print(f"  Avg ElasticNet test R^2 : {result_df['enet_test_r2'].mean():.4f}")
        print(f"  Avg XGBoost    test R^2 : {result_df['xgb_test_r2'].mean():.4f}")
        print(f"  Avg Lasso      test RMSE: {result_df['lasso_test_rmse'].mean():.4f}")
        print(f"  Avg ElasticNet test RMSE: {result_df['enet_test_rmse'].mean():.4f}")
        print(f"  Avg XGBoost    test RMSE: {result_df['xgb_test_rmse'].mean():.4f}")
        print()

        if save_summary_plot:
            plot_rolling_backtest_summary(
                result_df,
                save_path=str(out_dir / "rolling_backtest_summary.png"),
                show=show_plots,
            )

    return result_df


def plot_rolling_backtest_summary(
    result_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Line + bar chart summarising per-year R^2 and RMSE across folds."""
    years = result_df["test_year"].astype(int).values
    x = np.arange(len(years))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R^2
    ax = axes[0]
    ax.plot(x, result_df["lasso_test_r2"], "o-", label="Lasso", color=_CLR_LASSO)
    if "enet_test_r2" in result_df.columns:
        ax.plot(x, result_df["enet_test_r2"], "^-", label="Elastic Net", color=_CLR_ENET)
    ax.plot(x, result_df["xgb_test_r2"], "s-", label="XGBoost", color=_CLR_XGB)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.set_xlabel("Test year")
    ax.set_ylabel("Test R$^2$")
    ax.set_title("Rolling backtest: Test R$^2$ by year")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE
    ax = axes[1]
    n_models = 3 if "enet_test_rmse" in result_df.columns else 2
    w = 0.25
    offsets = np.linspace(-w, w, n_models)
    ax.bar(x + offsets[0], result_df["lasso_test_rmse"], w,
           label="Lasso", color=_CLR_LASSO, edgecolor="black", linewidth=0.4)
    if "enet_test_rmse" in result_df.columns:
        ax.bar(x + offsets[1], result_df["enet_test_rmse"], w,
               label="Elastic Net", color=_CLR_ENET, edgecolor="black", linewidth=0.4)
    ax.bar(x + offsets[-1], result_df["xgb_test_rmse"], w,
           label="XGBoost", color=_CLR_XGB, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.set_xlabel("Test year")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Rolling backtest: Test RMSE by year")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Expanding-window rolling backtest results", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ===================================================================
# Cross-split comparison plots (3 models)
# ===================================================================
def plot_split_test_r2_comparison(
    r2_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Grouped bar chart comparing Test R^2 across split strategies."""
    splits = list(r2_results.keys())
    model_names = list(r2_results[splits[0]].keys())
    colors = {
        "Lasso": _CLR_LASSO,
        "Elastic Net": _CLR_ENET,
        "XGBoost": _CLR_XGB,
    }

    x = np.arange(len(splits))
    n = len(model_names)
    w = 0.75 / n

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_groups = []
    for i, m in enumerate(model_names):
        vals = [r2_results[s][m] for s in splits]
        offset = (i - (n - 1) / 2) * w
        b = ax.bar(x + offset, vals, w, label=m,
                   color=colors.get(m, f"C{i}"), edgecolor="black", linewidth=0.5)
        bar_groups.append(b)

    for bg in bar_groups:
        for bar in bg:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.4f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Test R$^2$")
    ax.set_title("Out-of-Sample Test R$^2$ Across Data Split Strategies")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_split_test_rmse_comparison(
    rmse_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Grouped bar chart comparing Test RMSE across split strategies."""
    splits = list(rmse_results.keys())
    model_names = list(rmse_results[splits[0]].keys())
    colors = {
        "Lasso": _CLR_LASSO,
        "Elastic Net": _CLR_ENET,
        "XGBoost": _CLR_XGB,
    }

    x = np.arange(len(splits))
    n = len(model_names)
    w = 0.75 / n

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_groups = []
    for i, m in enumerate(model_names):
        vals = [rmse_results[s][m] for s in splits]
        offset = (i - (n - 1) / 2) * w
        b = ax.bar(x + offset, vals, w, label=m,
                   color=colors.get(m, f"C{i}"), edgecolor="black", linewidth=0.5)
        bar_groups.append(b)

    for bg in bar_groups:
        for bar in bg:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{bar.get_height():.4f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Test RMSE")
    ax.set_title("Out-of-Sample Test RMSE Across Data Split Strategies")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_all_splits_and_compare(
    df: pd.DataFrame,
    target_col: str = "win_rate",
    corr_threshold: float = 0.85,
    id_columns: Optional[List[str]] = None,
    random_state: int = 42,
    year_col: str = "year",
    train_years: Optional[List[int]] = None,
    test_years: Optional[List[int]] = None,
    min_train_years: int = 3,
    test_size: float = 0.2,
    use_time_series_cv: bool = True,
    show_plots: bool = True,
    plot_dir: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Run all three split modes and produce comparison bar charts
    for Lasso, Elastic Net, and XGBoost.

    Returns (r2_results, rmse_results) dicts.
    """
    out_dir = Path(plot_dir) if plot_dir else Path(".")
    common = dict(
        target_col=target_col, corr_threshold=corr_threshold,
        id_columns=id_columns, random_state=random_state,
        show_plots=False, save_extra_plots=False,
        use_time_series_cv=use_time_series_cv,
    )

    print("=" * 60)
    print("[Cross-split comparison] Running 3 split modes ...")
    print("=" * 60)
    print()

    # 1. random
    print(">>> split_mode = random")
    res_rand = run_full_pipeline(df, split_mode="random",
                                 test_size=test_size, **common)
    r_s2, r_e, r_s3 = res_rand["stage2"], res_rand["stage2_enet"], res_rand["stage3"]

    # 2. year_holdout
    print(">>> split_mode = year_holdout")
    res_hold = run_full_pipeline(df, split_mode="year_holdout",
                                 train_years=train_years,
                                 test_years=test_years,
                                 year_col=year_col,
                                 test_size=test_size, **common)
    h_s2, h_e, h_s3 = res_hold["stage2"], res_hold["stage2_enet"], res_hold["stage3"]

    # 3. rolling_year
    print(">>> split_mode = rolling_year")
    roll_df = run_full_pipeline(df, split_mode="rolling_year",
                                year_col=year_col,
                                min_train_years=min_train_years,
                                **common)

    r2_results = {
        "random": {
            "Lasso": r_s2.test_r2,
            "Elastic Net": r_e.test_r2,
            "XGBoost": r_s3.test_r2,
        },
        "year_holdout": {
            "Lasso": h_s2.test_r2,
            "Elastic Net": h_e.test_r2,
            "XGBoost": h_s3.test_r2,
        },
        "rolling_year": {
            "Lasso": float(roll_df["lasso_test_r2"].mean()),
            "Elastic Net": float(roll_df["enet_test_r2"].mean()),
            "XGBoost": float(roll_df["xgb_test_r2"].mean()),
        },
    }
    rmse_results = {
        "random": {
            "Lasso": r_s2.test_rmse,
            "Elastic Net": r_e.test_rmse,
            "XGBoost": r_s3.test_rmse,
        },
        "year_holdout": {
            "Lasso": h_s2.test_rmse,
            "Elastic Net": h_e.test_rmse,
            "XGBoost": h_s3.test_rmse,
        },
        "rolling_year": {
            "Lasso": float(roll_df["lasso_test_rmse"].mean()),
            "Elastic Net": float(roll_df["enet_test_rmse"].mean()),
            "XGBoost": float(roll_df["xgb_test_rmse"].mean()),
        },
    }

    print()
    print("=" * 60)
    print("[Cross-split comparison results]")
    print("=" * 60)
    for mode in r2_results:
        parts = []
        for m in r2_results[mode]:
            r2v = r2_results[mode][m]
            rmv = rmse_results[mode][m]
            parts.append(f"{m} R^2={r2v:.4f} RMSE={rmv:.4f}")
        print(f"  {mode:15s}  " + "  |  ".join(parts))
    print()

    plot_split_test_r2_comparison(
        r2_results,
        save_path=str(out_dir / "split_test_r2_comparison.png"),
        show=show_plots,
    )
    plot_split_test_rmse_comparison(
        rmse_results,
        save_path=str(out_dir / "split_test_rmse_comparison.png"),
        show=show_plots,
    )

    print(f"  Saved: {out_dir / 'split_test_r2_comparison.png'}")
    print(f"  Saved: {out_dir / 'split_test_rmse_comparison.png'}")
    print()

    return r2_results, rmse_results


# ===================================================================
# Extra diagnostic plots (3-model versions)
# ===================================================================
def _save_or_show(fig, save_path: Optional[str], show: bool) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_pipeline_flowchart(
    save_path: Optional[str] = None, show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7); ax.axis("off")

    box_s = dict(boxstyle="round,pad=0.4", facecolor="#E8EAF6", edgecolor="#3949AB", linewidth=1.5)
    spl_s = dict(boxstyle="round,pad=0.4", facecolor="#FFECB3", edgecolor="#F57F17", linewidth=1.5)
    enet_s = dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.5)
    eva_s = dict(boxstyle="round,pad=0.4", facecolor="#C8E6C9", edgecolor="#2E7D32", linewidth=1.5)

    nodes = [
        (1.0, 3.5, "Raw Data\n(2014-2025\nteam metrics)", box_s),
        (3.5, 3.5, "Year-based\nTrain / Test\nSplit", spl_s),
        (6.2, 5.5, "Stage 1\nPreprocessing\n(fit on train)", box_s),
        (6.2, 1.5, "Stage 1\nTransform\n(apply to test)", box_s),
        (9.2, 6.2, "Stage 2a\nLassoCV", box_s),
        (9.2, 4.8, "Stage 2b\nElasticNetCV", enet_s),
        (12.2, 5.5, "Stage 3\nXGBoost", box_s),
        (12.2, 1.5, "Evaluate\non Test Year(s)\n+ SHAP", eva_s),
    ]
    for x, y, txt, sty in nodes:
        ax.text(x, y, txt, ha="center", va="center", fontsize=8, fontweight="bold", bbox=sty)

    arrows = [
        (1.9, 3.5, 2.6, 3.5),
        (4.4, 4.0, 5.2, 5.5), (4.4, 3.0, 5.2, 1.5),
        (7.2, 5.8, 8.2, 6.2), (7.2, 5.2, 8.2, 4.8),
        (10.2, 6.0, 11.2, 5.7), (10.2, 5.1, 11.2, 5.3),
        (12.2, 4.8, 12.2, 2.2),
        (7.2, 1.5, 11.2, 1.5),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#37474F", lw=1.5))

    ax.text(8.0, 0.4, "Leakage-Free ML Pipeline (Year-Based Split)", fontsize=14,
            fontweight="bold", ha="center", va="center", color="#1A237E")
    ax.text(6.2, 3.5, "train years", fontsize=7, ha="center", color="#F57F17", style="italic")
    ax.text(6.2, 0.6, "test year(s) (transform only)", fontsize=7, ha="center",
            color="#F57F17", style="italic")
    _save_or_show(fig, save_path, show)


def plot_feature_reduction(
    n_raw: int, n_after_stage1: int, n_after_lasso: int,
    save_path: Optional[str] = None, show: bool = True,
) -> None:
    stages = ["Raw features", "After Stage 1\n(collinearity filter)", "After Stage 2\n(Lasso selection)"]
    counts = [n_raw, n_after_stage1, n_after_lasso]
    colors = ["#90A4AE", "#42A5F5", "#26A69A"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(stages, counts, color=colors, edgecolor="black", linewidth=0.5, width=0.55)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(cnt), ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("Number of features")
    ax.set_title("Feature reduction across pipeline stages")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_actual_vs_predicted(
    y_test: np.ndarray,
    lasso_pred: np.ndarray,
    enet_pred: np.ndarray,
    xgb_pred: np.ndarray,
    lasso_r2: float,
    enet_r2: float,
    xgb_r2: float,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    all_preds = np.concatenate([lasso_pred, enet_pred, xgb_pred])
    lo = min(np.min(y_test), np.min(all_preds)) - 0.03
    hi = max(np.max(y_test), np.max(all_preds)) + 0.03

    for ax, pred, name, r2, color in [
        (axes[0], lasso_pred, "Lasso", lasso_r2, _CLR_LASSO),
        (axes[1], enet_pred, "Elastic Net", enet_r2, _CLR_ENET),
        (axes[2], xgb_pred, "XGBoost", xgb_r2, _CLR_XGB),
    ]:
        ax.scatter(y_test, pred, alpha=0.75, edgecolors="black", linewidth=0.4,
                   color=color, s=50, zorder=3)
        ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1, zorder=2)
        ax.set_xlabel("Actual win rate"); ax.set_ylabel("Predicted win rate")
        ax.set_title(f"{name}  (Test R$^2$ = {r2:.4f})")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    fig.suptitle("Actual vs Predicted win rate on hold-out test set", fontsize=13, y=1.02)
    plt.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_three_model_r2_comparison(
    s2: Stage2Result,
    s2e: Stage2ElasticNetResult,
    s3: Stage3Result,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    labels = ["Train", "Test"]
    x = np.arange(len(labels))
    w = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w, [s2.train_r2, s2.test_r2], w, label="Lasso",
                color=_CLR_LASSO, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x, [s2e.train_r2, s2e.test_r2], w, label="Elastic Net",
                color=_CLR_ENET, edgecolor="black", linewidth=0.5)
    b3 = ax.bar(x + w, [s3.train_r2, s3.test_r2], w, label="XGBoost",
                color=_CLR_XGB, edgecolor="black", linewidth=0.5)
    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("R$^2$")
    ax.set_title("Model comparison: R$^2$ (Train vs Test)")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.1)
    ax.legend(); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_three_model_rmse_comparison(
    s2: Stage2Result,
    s2e: Stage2ElasticNetResult,
    s3: Stage3Result,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    labels = ["Train", "Test"]
    x = np.arange(len(labels))
    w = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w, [s2.train_rmse, s2.test_rmse], w, label="Lasso",
                color=_CLR_LASSO, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x, [s2e.train_rmse, s2e.test_rmse], w, label="Elastic Net",
                color=_CLR_ENET, edgecolor="black", linewidth=0.5)
    b3 = ax.bar(x + w, [s3.train_rmse, s3.test_rmse], w, label="XGBoost",
                color=_CLR_XGB, edgecolor="black", linewidth=0.5)
    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("Model comparison: RMSE (Train vs Test)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_residual_comparison(
    y_test: np.ndarray,
    lasso_pred: np.ndarray,
    enet_pred: np.ndarray,
    xgb_pred: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, pred, name, color in [
        (axes[0], lasso_pred, "Lasso", _CLR_LASSO),
        (axes[1], enet_pred, "Elastic Net", _CLR_ENET),
        (axes[2], xgb_pred, "XGBoost", _CLR_XGB),
    ]:
        resid = np.array(y_test) - np.array(pred)
        ax.scatter(pred, resid, alpha=0.75, edgecolors="black", linewidth=0.4,
                   color=color, s=50, zorder=3)
        ax.axhline(0, color="grey", linestyle="--", linewidth=1, zorder=2)
        ax.set_xlabel("Predicted win rate"); ax.set_ylabel("Residual (actual - predicted)")
        ax.set_title(f"{name} residuals on test set"); ax.grid(True, alpha=0.3)
    fig.suptitle("Residual analysis: Lasso vs Elastic Net vs XGBoost", fontsize=13, y=1.02)
    plt.tight_layout()
    _save_or_show(fig, save_path, show)


# ===================================================================
# Data loader
# ===================================================================
def load_metrics_from_csvs(
    directory: str, pattern: str = "*_metrics.csv",
) -> pd.DataFrame:
    import glob
    paths = sorted(glob.glob(str(Path(directory) / pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    frames = []
    for p in paths:
        part = pd.read_csv(p)
        stem = Path(p).stem
        year = "".join(filter(str.isdigit, stem))[:4]
        if year:
            part = part.copy()
            part["year"] = int(year)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


# ===================================================================
# Full pipeline entry point
# ===================================================================
def run_full_pipeline(
    df: pd.DataFrame,
    target_col: str = "win_rate",
    corr_threshold: float = 0.85,
    id_columns: Optional[List[str]] = None,
    random_state: int = 42,
    show_plots: bool = True,
    save_extra_plots: bool = False,
    plot_dir: Optional[str] = None,
    # --- split control ---
    split_mode: str = "year_holdout",
    test_size: float = 0.2,
    train_years: Optional[List[int]] = None,
    test_years: Optional[List[int]] = None,
    year_col: str = "year",
    min_train_years: int = 3,
    # --- internal CV inside LassoCV / ElasticNetCV / GridSearchCV ---
    use_time_series_cv: bool = True,
) -> Union[dict, pd.DataFrame]:
    """
    Execute the full leakage-free pipeline.

    Returns
    -------
    dict  when split_mode in ("random", "year_holdout")
        Keys: "stage1", "stage2", "stage2_enet", "stage3"
    pd.DataFrame  when split_mode == "rolling_year"
        One row per test-year fold with metrics for all three models.
    """
    valid_modes = ("random", "year_holdout", "rolling_year")
    if split_mode not in valid_modes:
        raise ValueError(f"split_mode must be one of {valid_modes}, got '{split_mode}'")

    out_dir = Path(plot_dir) if plot_dir else Path(".")

    target_col = resolve_target_column(df, target_col)
    df = df.copy()
    df = df.dropna(subset=[target_col])
    df = df.reset_index(drop=True)

    # ---- rolling_year delegates to its own function ----
    if split_mode == "rolling_year":
        return run_rolling_year_backtest(
            df, target_col=target_col, corr_threshold=corr_threshold,
            id_columns=id_columns, year_col=year_col,
            min_train_years=min_train_years, random_state=random_state,
            use_time_series_cv=use_time_series_cv,
            show_plots=show_plots,
            save_summary_plot=save_extra_plots,
            plot_dir=str(out_dir),
        )

    # ---- determine train/test indices ----
    if split_mode == "year_holdout":
        if train_years is None or test_years is None:
            all_years = sorted(df[year_col].unique())
            n_test = max(1, int(len(all_years) * test_size))
            train_years = list(all_years[:-n_test])
            test_years = list(all_years[-n_test:])
            print(f"[year_holdout] Auto-split: train {train_years}, test {test_years}\n")

        train_idx, test_idx = get_year_based_split(
            df, train_years, test_years, year_col,
        )
        print(f"[year_holdout] Train years: {train_years}")
        print(f"[year_holdout] Test  years: {test_years}")
        print(f"[year_holdout] Train rows : {len(train_idx)}  |  Test rows: {len(test_idx)}")
        print()

    else:  # "random"
        train_idx, test_idx = _random_split_indices(
            len(df), test_size, random_state,
        )

    # ---- run single fold (now includes Elastic Net) ----
    res = _run_single_fold(
        df, train_idx, test_idx,
        target_col=target_col, corr_threshold=corr_threshold,
        id_columns=id_columns, random_state=random_state,
        use_time_series_cv=use_time_series_cv,
        show_plots=show_plots,
        save_extra_plots=save_extra_plots, out_dir=out_dir,
    )

    s1 = res["stage1"]
    s2 = res["stage2"]
    s2e = res["stage2_enet"]
    s3 = res["stage3"]

    # ---- extra diagnostic plots (3-model versions) ----
    if save_extra_plots:
        y_te = np.array(s1.y_test)
        plot_pipeline_flowchart(save_path=str(out_dir / "pipeline_flowchart.png"), show=show_plots)
        plot_feature_reduction(
            n_raw=s1.n_features_raw,
            n_after_stage1=s1.artifacts.n_features_after,
            n_after_lasso=len(s2.selected_features),
            save_path=str(out_dir / "feature_reduction.png"), show=show_plots,
        )
        plot_actual_vs_predicted(
            y_test=y_te,
            lasso_pred=s2.test_pred, enet_pred=s2e.test_pred, xgb_pred=s3.test_pred,
            lasso_r2=s2.test_r2, enet_r2=s2e.test_r2, xgb_r2=s3.test_r2,
            save_path=str(out_dir / "actual_vs_predicted_comparison.png"), show=show_plots,
        )
        plot_three_model_r2_comparison(
            s2, s2e, s3,
            save_path=str(out_dir / "model_performance_comparison.png"), show=show_plots,
        )
        plot_three_model_rmse_comparison(
            s2, s2e, s3,
            save_path=str(out_dir / "model_rmse_comparison.png"), show=show_plots,
        )
        plot_residual_comparison(
            y_test=y_te,
            lasso_pred=s2.test_pred, enet_pred=s2e.test_pred, xgb_pred=s3.test_pred,
            save_path=str(out_dir / "residual_comparison.png"), show=show_plots,
        )
        print("=" * 60)
        print("[Extra plots saved]")
        print("=" * 60)
        for name in [
            "pipeline_flowchart.png", "feature_reduction.png",
            "actual_vs_predicted_comparison.png",
            "model_performance_comparison.png", "model_rmse_comparison.png",
            "residual_comparison.png",
            "lasso_coefficients.png", "elasticnet_coefficients.png",
            "shap_summary.png",
        ]:
            print(f"  {out_dir / name}")
        print()

    return res


# ===================================================================
# CLI entry point
# ===================================================================
if __name__ == "__main__":
    import sys

    _here = Path(__file__).resolve().parent
    repo_root = _here.parent
    plot_dir = repo_root / "outputs" / "figures"
    try:
        from data_layout import resolve_processed_data_dir

        csv_dir = resolve_processed_data_dir(repo_root)
    except FileNotFoundError:
        print("No *_metrics.csv found. Add licensed data under data/processed/ ")
        print("or use bundled sample_data/processed/ (see data/README.md).")
        print("  python main.py")
        sys.exit(1)
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading data from: {csv_dir}\n")
    data = load_metrics_from_csvs(str(csv_dir))
    if "Team" in data.columns and "team" not in data.columns:
        data = data.rename(columns={"Team": "team"})

    run_full_pipeline(
        data,
        target_col="win_rate",
        split_mode="year_holdout",
        save_extra_plots=True,
        plot_dir=str(plot_dir),
        show_plots=False,
    )
