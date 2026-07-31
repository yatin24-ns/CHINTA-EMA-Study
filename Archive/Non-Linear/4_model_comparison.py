"""
=============================================================================
PIPELINE STAGE 4: CORE MODEL COMPARISON & UNIFIED REPORT
=============================================================================
Compares the 4 core linear models:
  - Frequentist Linear (Original vs Imputed)
  - Bayesian Linear (Original vs Imputed)
  
Generates a unified text report `model_comparison_report.txt` focusing on
model fit metrics (AIC, BIC, LogLik, LOO-ELPD) and coefficient estimates
with uncertainty bounds (95% CI for frequentist, 95% HDI for Bayesian).
=============================================================================
"""

import os
import json
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from stats_utils import BASE_DIR, DATASETS

REPORT_OUTPUT = os.path.join(BASE_DIR, "model_comparison_report.txt")
FL_JSON = os.path.join(BASE_DIR, "frequentist_linear_metrics.json")
BL_JSON = os.path.join(BASE_DIR, "bayesian_linear_metrics.json")

def load_json_metrics(path):
    """Load metrics from a JSON file."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def _scalar(v):
    """Extract scalar from a value that may be a dict (e.g. {mean, sd, min, max}) or already a float."""
    if isinstance(v, dict): return v.get('mean', np.nan)
    return v if v is not None else np.nan

def safe_float(val, precision=4):
    """Safely convert value to string with specific precision."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    try:
        return f"{float(val):.{precision}f}"
    except (TypeError, ValueError):
        return str(val)

def run_comparison():
    print("=" * 70)
    print("  PIPELINE STAGE 4: CORE MODEL COMPARISON")
    print("=" * 70)
    
    fl_metrics = load_json_metrics(FL_JSON)
    bl_metrics = load_json_metrics(BL_JSON)
    
    report = []
    report.append("=" * 90)
    report.append("  CORE MODEL COMPARISON REPORT (4 MODELS)")
    report.append("=" * 90)
    
    report.append("\n  --- OVERVIEW ---")
    report.append("  This report compares the 4 core linear mixed models to assess the robustness")
    report.append("  of the fixed effects across estimation paradigms (Frequentist vs Bayesian)")
    report.append("  and missing data strategies (Complete-Cases vs m=20 Passive Imputation).")
    report.append("\n  * Primary Dataset (Original): 43 complete cases across 18 participants.")
    report.append("  * Imputed Dataset (Passive): m=20 imputations of the full 78-row dataset.")
    
    # ---------------------------------------------------------------------
    # 1. MODEL FIT METRICS
    # ---------------------------------------------------------------------
    report.append("\n" + "-" * 90)
    report.append("  SECTION 1: MODEL FIT METRICS")
    report.append("-" * 90)
    
    # Frequentist
    report.append("\n  [ Frequentist Models (Linear Mixed Models) ]")
    report.append(f"  {'Model':<25} | {'N (Mean)':<10} | {'AIC':<10} | {'BIC':<10} | {'LogLik':<10} | {'ICC':<10}")
    report.append("  " + "-" * 75)
    
    for key, label in [("Linear_2L_original", "Freq (Original)"), ("Linear_2L_imputed", "Freq (Imputed)")]:
        m = fl_metrics.get(key, {})
        n_obs = int(np.round(_scalar(m.get("N_obs_mean", 0))))
        aic = safe_float(_scalar(m.get("AIC_ML")), 2)
        bic = safe_float(_scalar(m.get("BIC_ML")), 2)
        ll = safe_float(_scalar(m.get("LogLikelihood_ML")), 2)
        icc = safe_float(m.get("ICC_mean_REML"), 3)
        report.append(f"  {label:<25} | {n_obs:<10} | {aic:<10} | {bic:<10} | {ll:<10} | {icc:<10}")
        
    report.append("\n  Note: AIC/BIC values are NOT directly comparable between Original and Imputed")
    report.append("  because the effective sample size (N) differs fundamentally.")

    # Bayesian
    report.append("\n  [ Bayesian Models (MCMC) ]")
    report.append(f"  {'Model':<25} | {'N (Mean)':<10} | {'LOO-ELPD':<15} | {'R-hat Max':<10} | {'Converged':<10}")
    report.append("  " + "-" * 78)
    
    for key, label in [("Linear_2L_original", "Bayes (Original)"), ("Linear_2L_imputed", "Bayes (Imputed)")]:
        m = bl_metrics.get(key, {})
        # N_obs is not explicitly saved in Bayesian JSON in the same way, but we know it
        n_obs = 43 if key == "Linear_2L_original" else 78
        loo = m.get("LOO_elpd_mean")
        loo_sd = m.get("LOO_elpd_sd")
        loo_str = f"{safe_float(loo, 2)} (±{safe_float(loo_sd, 2)})" if loo is not None else "N/A"
        rhat = safe_float(m.get("R_hat_max"), 3)
        conv = "Yes" if m.get("Converged", False) else "No"
        report.append(f"  {label:<25} | {n_obs:<10} | {loo_str:<15} | {rhat:<10} | {conv:<10}")

    # ---------------------------------------------------------------------
    # 2. PARAMETER ESTIMATES COMPARISON
    # ---------------------------------------------------------------------
    report.append("\n" + "-" * 90)
    report.append("  SECTION 2: PARAMETER ESTIMATES & UNCERTAINTY")
    report.append("-" * 90)
    report.append("  Freq bounds = 95% CI (Estimate ± 1.96*SE)")
    report.append("  Bayes bounds = 95% HDI (Highest Density Interval from posterior draws)")
    
    # Extract params
    params = {}
    skip = {'Intercept', 'Group Var', 'sigma', 'sigma_log__'}
    
    # Load Freq
    for key, label in [("Linear_2L_original", "Freq (Original)"), ("Linear_2L_imputed", "Freq (Imputed)")]:
        for p, v in fl_metrics.get(key, {}).get("Parameters", {}).items():
            if p in skip: continue
            if p not in params: params[p] = {}
            est = v["Estimate"]
            se = v["SE"]
            lb = est - 1.96 * se
            ub = est + 1.96 * se
            params[p][label] = f"{est: .3f} [{lb: .3f}, {ub: .3f}]"
            
    # Load Bayes
    for key, label in [("Linear_2L_original", "Bayes (Original)"), ("Linear_2L_imputed", "Bayes (Imputed)")]:
        blk = bl_metrics.get(key, {})
        b_params = {k[5:-5] for k in blk if k.startswith('post_') and k.endswith('_mean')}
        for p in b_params:
            if p in skip: continue
            if p not in params: params[p] = {}
            est = blk.get(f"post_{p}_mean")
            lb = blk.get(f"post_{p}_hdi_lb")
            ub = blk.get(f"post_{p}_hdi_ub")
            if est is not None and lb is not None and ub is not None:
                params[p][label] = f"{est: .3f} [{lb: .3f}, {ub: .3f}]"
            else:
                params[p][label] = "N/A"
                
    # Format Table
    models = ["Freq (Original)", "Freq (Imputed)", "Bayes (Original)", "Bayes (Imputed)"]
    header = f"  {'Parameter':<20} | {'Freq (Original)':<24} | {'Freq (Imputed)':<24} | {'Bayes (Original)':<24} | {'Bayes (Imputed)':<24}"
    report.append("\n" + header)
    report.append("  " + "-" * 125)
    
    # Sort params (WP first, then BP)
    wp = [p for p in params.keys() if 'WP' in p or 'day' in p or 'weekend' in p]
    bp = [p for p in params.keys() if p not in wp]
    ordered_params = sorted(wp) + sorted(bp)
    
    for p in ordered_params:
        row = f"  {p:<20} | "
        row += " | ".join([f"{params[p].get(m, 'N/A'):<24}" for m in models])
        report.append(row)
        
    report.append("\n" + "-" * 90)
    report.append("  END OF REPORT")
    report.append("-" * 90)

    # Save report
    print(f"\n  Saving core comparison report to {REPORT_OUTPUT}...")
    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print(f"  [+] Report saved.")

if __name__ == "__main__":
    run_comparison()
