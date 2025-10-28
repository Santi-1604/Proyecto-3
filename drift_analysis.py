#!/usr/bin/env python
# coding: utf-8

# # Nueva Prueba

# In[22]:


import os
import shutil
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="muted", font_scale=1.0)
plt.rcParams['figure.figsize'] = (12, 5)

# ---------------------------
# Detección con ventana deslizante y construcción de resumen
# ---------------------------
def _find_drift_start_index(
    df: pd.DataFrame,
    feat: str,
    train_end: int,
    alpha: float = 0.01,
    window: int = 100,
    step: int = 10,
    consecutive: int = 2,
) -> Optional[int]:
    """Devuelve el índice de posición donde comienza el drift (fin de la ventana). None si no se encuentra."""
    baseline = df.iloc[:train_end][feat].dropna().values
    if baseline.size < 2:
        return None

    consec = 0
    n = len(df)
    last_start = max(train_end, n - window + 1)
    for start in range(train_end, last_start + 1, step):
        seg_vals = df.iloc[start:start + window][feat].dropna().values
        if seg_vals.size < 2:
            consec = 0
            continue
        p = ks_2samp(baseline, seg_vals, alternative="two-sided", mode="auto").pvalue
        if np.isfinite(p) and p < alpha:
            consec += 1
            if consec >= consecutive:
                return min(start + window - 1, n - 1)
        else:
            consec = 0
    return None

def _flag_title(p_test: float, p_val: float, alpha: float) -> str:
    """Genera título indicando si se detectó drift"""
    flags = []
    if np.isfinite(p_test) and p_test < alpha:
        flags.append("DRIFT (train↔test)")
    if np.isfinite(p_val) and p_val < alpha:
        flags.append("DRIFT (train↔val)")
    return " | ".join(flags) if flags else "No hay drift significativo"

def _build_summary_md(stats: pd.DataFrame, alpha: float = 0.01, top_k: int = 5) -> str:
    """Construye resumen Markdown con las top features más afectadas por drift"""
    s = stats.copy()
    s["min_p"] = s[["ks_p_train_vs_test", "ks_p_train_vs_val"]].min(axis=1)
    top = s.sort_values("min_p", ascending=True).head(top_k)
    lines = ["# Top Features con Drift\n"]
    for _, r in top.iterrows():
        reasons = []
        if np.isfinite(r["ks_p_train_vs_test"]) and r["ks_p_train_vs_test"] < alpha:
            reasons.append(f"train→test: Δμ={r['mean_delta_test']:.4g} ({r['mean_pct_change_test']:.2f}%), Var×={r['var_ratio_test']:.3g}")
        if np.isfinite(r["ks_p_train_vs_val"]) and r["ks_p_train_vs_val"] < alpha:
            reasons.append(f"train→val: Δμ={r['mean_delta_val']:.4g} ({r['mean_pct_change_val']:.2f}%), Var×={r['var_ratio_val']:.3g}")
        reason_txt = "; ".join(reasons) if reasons else "p-valor más bajo pero sin cambios claros en media/varianza"
        lines.append(f"- **{r['feature']}** — {reason_txt}")
    lines.append("\nPistas de interpretación: |Δμ| grande sugiere cambio de ubicación; Var× lejos de 1 implica cambio de escala.")
    return "\n".join(lines)

# ---------------------------
# Función de plot (inline)
# ---------------------------
def _plot_feature_distributions_inline(
    df: pd.DataFrame,
    feat: str,
    train_end: int,
    test_end: int,
    title_suffix: str,
    alpha: float = 0.01,
    scan_window: int = 100,
    scan_step: int = 10,
    drift_consecutive: int = 2,
    show_kde: bool = True,
):
    """Muestra timeline + distribución (KDE) para una feature. Marca inicio del drift si se encuentra."""
    # Preparar arrays
    series = df[feat]
    train_vals = df.iloc[:train_end][feat].dropna().values
    test_vals = df.iloc[train_end:test_end][feat].dropna().values
    val_vals = df.iloc[test_end:][feat].dropna().values

    # Timeline (arriba)
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[1,1]}, figsize=(12,8))
    ax1.plot(df.index, series.values, linewidth=0.8)
    ax1.set_title(f"{feat} — {title_suffix}")
    ax1.set_ylabel("Valor")
    # sombrear periodos
    ax1.axvspan(df.index[0], df.index[train_end-1], color='green', alpha=0.08, label='Train')
    ax1.axvspan(df.index[train_end], df.index[test_end-1], color='orange', alpha=0.08, label='Test')
    ax1.axvspan(df.index[test_end], df.index[-1], color='red', alpha=0.08, label='Val')
    ax1.legend(loc='upper left')

    # Encontrar inicio del drift y marcarlo
    drift_pos = _find_drift_start_index(df, feat, train_end, alpha=alpha, window=scan_window, step=scan_step, consecutive=drift_consecutive)
    if drift_pos is not None:
        x_drift = df.index[drift_pos]
        ax1.axvline(x=x_drift, color='purple', linestyle='--', linewidth=1.5)
        ax1.text(x_drift, ax1.get_ylim()[1], "Inicio Drift", rotation=90, va='top', ha='right', fontsize=9, color='purple')

    # Comparación de distribución (abajo)
    # Usar curvas KDE para comparabilidad suave (histograma si KDE falla)
    def safe_kde_plot(data, axis, label, color):
        try:
            if len(data) > 1:
                sns.kdeplot(data, ax=axis, label=label, fill=False, bw_method='scott', linewidth=1.5, color=color)
        except Exception:
            axis.hist(data, bins=40, alpha=0.3, label=label, color=color)

    colors = {'Train':'green','Test':'orange','Val':'red'}
    if show_kde:
        safe_kde_plot(train_vals, ax2, 'Train', colors['Train'])
        safe_kde_plot(test_vals, ax2, 'Test', colors['Test'])
        safe_kde_plot(val_vals, ax2, 'Val', colors['Val'])
    else:
        ax2.hist(train_vals, bins=40, density=True, alpha=0.3, label='Train', color=colors['Train'])
        ax2.hist(test_vals, bins=40, density=True, alpha=0.3, label='Test', color=colors['Test'])
        ax2.hist(val_vals, bins=40, density=True, alpha=0.3, label='Val', color=colors['Val'])

    ax2.set_xlabel("Valor")
    ax2.set_ylabel("Densidad")
    ax2.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------
# API principal: run_drift_report (visualización inline + guardado opcional)
# ---------------------------
def run_drift_report(
    df: pd.DataFrame,
    target_col: str,
    train_frac: float = 0.6,
    test_frac: float = 0.2,
    alpha: float = 0.01,
    out_dir: str = "drift_report",
    features: List[str] | None = None,
    scan_window: int = 100,
    scan_step: int = 10,
    drift_consecutive: int = 2,
    save_outputs: bool = True,   # si False no guarda CSV ni SUMMARY.md
    show_plots: bool = True,     # si True muestra plots inline
) -> pd.DataFrame:
    """
    Ejecuta análisis de drift y retorna DataFrame de estadísticas.
    Muestra gráficas en pantalla (si show_plots=True) y opcionalmente
    guarda drift_stats.csv y SUMMARY.md en out_dir (si save_outputs=True).
    """
    # Preparar carpeta de salida
    if save_outputs:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    # Validaciones básicas
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df debe ser un pandas DataFrame")
    n = len(df)
    if n < 3:
        raise ValueError("df demasiado pequeño para dividir en train/test/val")

    # División cronológica
    train_end = int(train_frac * n)
    test_end = int((train_frac + test_frac) * n)
    if train_end < 2 or test_end <= train_end:
        raise ValueError("Parámetros de split producen particiones inválidas")

    # Seleccionar features (numéricas excluyendo target)
    if features is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric_cols if c != target_col]
    else:
        # filtrar solo columnas existentes y numéricas
        features = [c for c in features if c in df.columns and np.issubdtype(df[c].dtype, np.number)]

    if len(features) == 0:
        raise ValueError("No se encontraron features numéricas para evaluar drift.")

    # Construir periodos
    periods = {
        "train": df.iloc[:train_end].copy(),
        "test": df.iloc[train_end:test_end].copy(),
        "val": df.iloc[test_end:].copy(),
    }

    rows = []
    for feat in features:
        train_vals = periods["train"][feat].dropna().values
        test_vals = periods["test"][feat].dropna().values
        val_vals = periods["val"][feat].dropna().values

        # KS tests (train vs test, train vs val)
        try:
            ks_tt = ks_2samp(train_vals, test_vals, alternative="two-sided", mode="auto")
        except Exception:
            ks_tt = type("obj", (), {"pvalue": np.nan, "statistic": np.nan})
        try:
            ks_tv = ks_2samp(train_vals, val_vals, alternative="two-sided", mode="auto")
        except Exception:
            ks_tv = type("obj", (), {"pvalue": np.nan, "statistic": np.nan})

        # Describir los cambios
        def describe_change(ref: np.ndarray, comp: np.ndarray) -> Tuple[float, float, float]:
            if len(ref) == 0 or len(comp) == 0:
                return np.nan, np.nan, np.nan
            mu_ref, mu_comp = np.nanmean(ref), np.nanmean(comp)
            sd_ref, sd_comp = np.nanstd(ref), np.nanstd(comp)
            delta_mu = mu_comp - mu_ref
            pct_mu = (delta_mu / (abs(mu_ref) + 1e-9)) * 100.0
            var_ratio = (sd_comp**2) / (sd_ref**2 + 1e-12)
            return delta_mu, pct_mu, var_ratio

        dmu_tst, pmu_tst, vr_tst = describe_change(train_vals, test_vals)
        dmu_val, pmu_val, vr_val = describe_change(train_vals, val_vals)

        row = {
            "feature": feat,
            "ks_p_train_vs_test": ks_tt.pvalue,
            "ks_stat_train_vs_test": ks_tt.statistic,
            "ks_p_train_vs_val": ks_tv.pvalue,
            "ks_stat_train_vs_val": ks_tv.statistic,
            "drift_vs_test": bool(np.isfinite(ks_tt.pvalue) and ks_tt.pvalue < alpha),
            "drift_vs_val": bool(np.isfinite(ks_tv.pvalue) and ks_tv.pvalue < alpha),
            "mean_delta_test": dmu_tst,
            "mean_pct_change_test": pmu_tst,
            "var_ratio_test": vr_tst,
            "mean_delta_val": dmu_val,
            "mean_pct_change_val": pmu_val,
            "var_ratio_val": vr_val,
        }
        rows.append(row)

        title_suffix = _flag_title(ks_tt.pvalue, ks_tv.pvalue, alpha)
        if show_plots:
            _plot_feature_distributions_inline(
                df=df,
                feat=feat,
                train_end=train_end,
                test_end=test_end,
                title_suffix=title_suffix,
                alpha=alpha,
                scan_window=scan_window,
                scan_step=scan_step,
                drift_consecutive=drift_consecutive,
            )

    stats = pd.DataFrame(rows).sort_values(
        by=["drift_vs_test", "drift_vs_val", "ks_p_train_vs_test", "ks_p_train_vs_val"],
        ascending=[False, False, True, True]
    ).reset_index(drop=True)

    if save_outputs:
        stats_path = os.path.join(out_dir, "drift_stats.csv")
        stats.to_csv(stats_path, index=False)
        summary_md = _build_summary_md(stats, alpha=alpha)
        with open(os.path.join(out_dir, "SUMMARY.md"), "w", encoding="utf-8") as f:
            f.write(summary_md)

    print(f"Análisis de drift completado. {len(stats)} features analizadas.")
    if save_outputs:
        print(f"Resultados guardados en: {os.path.abspath(out_dir)}")

    return stats

# =============================================================
# Ejemplo rápido (simulación)
# =============================================================
if __name__ == "__main__":
    # Simular datos con drift
    np.random.seed(0)
    n = 2500
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    df_sim = pd.DataFrame({
        "feature1": np.random.normal(0,1,n),
        "feature2": np.concatenate([np.random.normal(0,1,int(n*0.6)), np.random.normal(1.5,1,int(n*0.4))]),
        "feature3": np.concatenate([np.random.normal(0,1,int(n*0.6)), np.random.normal(0,2,int(n*0.4))]),
        "feature4": np.random.normal(0,1,n),
        "target": np.random.choice([0,1], n)
    }, index=idx)

    stats = run_drift_report(df_sim, target_col="target", alpha=0.05, scan_window=150, scan_step=25, drift_consecutive=2, save_outputs=True, show_plots=True)
    display(stats.head())

