"""
=============================================================================
PIPELINE STAGE 4: FREQUENTIST NON-LINEAR MIXED MODELS
=============================================================================
Implements the piecewise spline + log-leisure non-linear architecture
using MLE estimation (statsmodels).

Key Non-Linear Features:
  1. Asymmetric Workload Threshold (Piecewise Spline):
     - Workload_Baseline_WP = min(0, Workload_WP_z)  → normal/light days
     - Workload_Crunch_WP   = max(0, Workload_WP_z)  → tipping point
     These are orthogonal by construction (one <= 0, other ≥ 0),
     eliminating multicollinearity.
     
  2. Log-Leisure (Diminishing Returns):
     - Leisure_Log = log(S3_Leisure_Mean + 1)  → taken BEFORE centering
     - This captures the psychological law of diminishing marginal returns:
       the first hour of leisure is more restorative than the fifth.
       
  3. Differential Interactions:
     - Interaction_Normal = Workload_Baseline × Leisure_Log_WP_z  → shielding effect
     - Interaction_Crunch = Workload_Crunch × Leisure_Log_WP_z    → shield-shatter test

Mathematical Architecture:
  Level 1 (Within-Person):
    Y_ij = β_0j + β_1j(Workload_Baseline) + β_2j(Workload_Crunch) 
         + β_3j(Leisure_Log_WP) + β_4j(Interaction_Normal) 
         + β_5j(Interaction_Crunch) + β_6j(Sleep_WP) 
         + β_7j(Workload_SD) + TimeControls + e_ij

  Level 2 (Between-Person):
    β_0j = γ_00 + γ_01(Workload_BP) + γ_02(Leisure_Log_BP) 
         + γ_03(Sleep_BP) + γ_04(Trait_Fatigue_z) + u_0j

Data Integration:
  - Runs on original_complete_cases.xlsx (Baseline)
  - Runs on imputed_processed_ema.parquet (m=5) and applies Rubin's Rules
=============================================================================
"""

from stats_utils import BASE_DIR, DATASETS, apply_rubins_rules
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats
import warnings
import json
import os

warnings.filterwarnings("ignore")

SUMMARY_OUTPUT = os.path.join(BASE_DIR, "frequentist_nonlinear_summary.txt")
METRICS_OUTPUT = os.path.join(BASE_DIR, "frequentist_nonlinear_metrics.json")


def prepare_nonlinear_data(df):
    """
    Prepares data for non-linear models.
    Computes piecewise spline and log transforms if not already present.
    """
    required_cols = [
        'S2_Fatigue_Mean_z', 'Participant_No',
        'Workload_BP_z', 'Sleep_BP_z', 'Trait_Fatigue_z',
        'day_number_z', 'is_week_onset', 'is_week_finish', 'is_weekend',
        'Sleep_WP_z',
    ]
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  [WARNING] Dropping dataset: Missing required columns: {missing}")
        return None
    
    if 'Workload_Baseline_WP' not in df.columns:
        if 'Workload_WP_z' in df.columns:
            df['Workload_Baseline_WP'] = np.minimum(0, df['Workload_WP_z'])
            df['Workload_Crunch_WP'] = np.maximum(0, df['Workload_WP_z'])
        else:
            return None
    
    if 'Leisure_Log_WP_z' not in df.columns:
        if 'S3_Leisure_Mean' in df.columns:
            df['Leisure_Log'] = np.log1p(df['S3_Leisure_Mean'])
            df['Leisure_Log_BP'] = df.groupby('Participant_No')['Leisure_Log'].transform('mean')
            df['Leisure_Log_WP'] = df['Leisure_Log'] - df['Leisure_Log_BP']
            for s in ['WP', 'BP']:
                col = f'Leisure_Log_{s}'
                m, sd = df[col].mean(), df[col].std()
                df[f'{col}_z'] = (df[col] - m) / sd if sd > 0 else 0.0
        else:
            return None
    
    if 'Interaction_Normal' not in df.columns:
        df['Interaction_Normal'] = df['Workload_Baseline_WP'] * df['Leisure_Log_WP_z']
        df['Interaction_Crunch'] = df['Workload_Crunch_WP'] * df['Leisure_Log_WP_z']
    
    df = df.dropna(subset=['S2_Fatigue_Mean_z']).copy()
    
    nl_predictors = [
        'Workload_Baseline_WP', 'Workload_Crunch_WP', 'Leisure_Log_WP_z',
        'Interaction_Normal', 'Interaction_Crunch', 'Sleep_WP_z',
        'Workload_BP_z', 'Leisure_Log_BP_z',
        'Sleep_BP_z', 'Trait_Fatigue_z', 'day_number_z',
    ]
    nl_predictors = [c for c in nl_predictors if c in df.columns]
    df = df.dropna(subset=nl_predictors).copy()
    
    return df


def run_frequentist_nonlinear_models():
    print("=" * 70)
    print("  PIPELINE STAGE 4: FREQUENTIST NON-LINEAR MODELS")
    print("=" * 70)
    
    nonlinear_formula = (
        "S2_Fatigue_Mean_z ~ "
        "Workload_Baseline_WP + Workload_Crunch_WP + "
        "Leisure_Log_WP_z + "
        "Interaction_Normal + Interaction_Crunch + "
        "Sleep_WP_z + "
        "Workload_BP_z + Leisure_Log_BP_z + Sleep_BP_z + "
        "Trait_Fatigue_z + day_number_z + is_week_onset + is_week_finish + is_weekend"
    )
    
    all_summaries = []
    all_metrics = {}
    
    # Run on original and imputed_passive datasets (passive is correct for interactions)
    datasets_to_check = {
        "original": DATASETS["original"],
        "imputed": DATASETS["imputed_passive"]
    }
    
    for dataset_name, dataset_path in datasets_to_check.items():
        if not os.path.exists(dataset_path):
            continue
        
        print(f"\n{'-' * 70}\n  DATASET: {dataset_name.upper()}\n{'-' * 70}")
        
        if dataset_name == "original":
            df = prepare_nonlinear_data(pd.read_excel(dataset_path))
            datasets_to_run = [(1, df)]
        else:
            df_full = pd.read_parquet(dataset_path)
            datasets_to_run = []
            for imp in sorted(df_full['.imp'].unique()):
                datasets_to_run.append((imp, prepare_nonlinear_data(df_full[df_full['.imp'] == imp])))
                
        # Fit 2L NonLinear Model
        model_name = f"NonLinear_2L_{dataset_name}"
        print(f"\n  [{model_name}] Fitting...")
        
        ml_stats = {"aic": [], "bic": [], "llf": [], "conv": [], "r2_m": [], "r2_c": []}
        reml_stats = {"conv": []}
        rs_stats = {"aic": [], "bic": []}
        
        ests_ml = []
        bse_ml = []
        ests_reml = []
        bse_reml = []
        n_obs_list = []
        n_participants_list = []
        iccs_ml = []
        iccs_reml = []
        failed = False
        
        for imp, df_imp in datasets_to_run:
            if df_imp is None or len(df_imp) == 0:
                failed = True; break
            try:
                # 1. Fit ML model
                model_ml = smf.mixedlm(nonlinear_formula, df_imp, groups=df_imp["Participant_No"])
                try:
                    res_ml = model_ml.fit(reml=False, method='lbfgs')
                except Exception:
                    res_ml = model_ml.fit(reml=False, method='bfgs')
                    
                ml_stats["aic"].append(res_ml.aic)
                ml_stats["bic"].append(res_ml.bic)
                ml_stats["llf"].append(res_ml.llf)
                ml_stats["conv"].append(res_ml.converged)
                
                ests_ml.append(res_ml.fe_params.to_dict())
                bse_ml.append(res_ml.bse_fe.to_dict())
                n_obs_list.append(res_ml.nobs)
                n_participants_list.append(df_imp["Participant_No"].nunique())
                
                if hasattr(res_ml, 'cov_re') and res_ml.cov_re.size > 0 and hasattr(res_ml, 'scale'):
                    var_u = float(res_ml.cov_re.iloc[0, 0])
                    var_e = float(res_ml.scale)
                    icc_ml = var_u / (var_u + var_e) if (var_u + var_e) > 0 else 0
                    iccs_ml.append(icc_ml)
                    
                    var_f = np.var(res_ml.predict(df_imp))
                    tot_var = var_f + var_u + var_e
                    if tot_var > 0:
                        ml_stats["r2_m"].append(var_f / tot_var)
                        ml_stats["r2_c"].append((var_f + var_u) / tot_var)
                        
                # 1.5 Fit Random Slope model for comparison
                model_rs = smf.mixedlm(nonlinear_formula, df_imp, groups=df_imp["Participant_No"], re_formula="~Workload_Baseline_WP + Workload_Crunch_WP + Leisure_Log_WP_z")
                try:
                    res_rs = model_rs.fit(reml=False, method='lbfgs')
                    rs_stats["aic"].append(res_rs.aic)
                    rs_stats["bic"].append(res_rs.bic)
                except Exception:
                    pass
                    
                # 2. Fit REML model
                model_reml = smf.mixedlm(nonlinear_formula, df_imp, groups=df_imp["Participant_No"])
                try:
                    res_reml = model_reml.fit(reml=True, method='lbfgs')
                except Exception:
                    res_reml = model_reml.fit(reml=True, method='bfgs')
                    
                reml_stats["conv"].append(res_reml.converged)
                ests_reml.append(res_reml.fe_params.to_dict())
                bse_reml.append(res_reml.bse_fe.to_dict())
                
                if hasattr(res_reml, 'cov_re') and res_reml.cov_re.size > 0 and hasattr(res_reml, 'scale'):
                    icc_reml = float(res_reml.cov_re.iloc[0, 0]) / (float(res_reml.cov_re.iloc[0, 0]) + float(res_reml.scale))
                    iccs_reml.append(icc_reml)
                    
            except Exception as e:
                print(f"  [ERROR] Imputation {imp} failed: {e}")
                continue
                
        if len(ests_ml) == 0:
            all_metrics[model_name] = {"Converged": False, "Dataset": dataset_name}
            print(f"  [WARNING] Estimation failed for {model_name}. Skipping pooling.")
            continue
            
        level_mapping = {
            "Intercept": 2,
            "Workload_Baseline_WP": 1,
            "Workload_Crunch_WP": 1,
            "Leisure_Log_WP_z": 1,
            "Interaction_Normal": 1,
            "Interaction_Crunch": 1,
            "Sleep_WP_z": 1,

            "Workload_BP_z": 2,
            "Leisure_Log_BP_z": 2,
            "Sleep_BP_z": 2,
            "Trait_Fatigue_z": 2,
            "day_number_z": 1,
            "is_week_onset": 1,
            "is_week_finish": 1,
            "is_weekend": 1
        }
        
        pooled_ml = apply_rubins_rules(ests_ml, bse_ml, n_obs_list, n_participants_list, level_mapping=level_mapping)
        pooled_reml = apply_rubins_rules(ests_reml, bse_reml, n_obs_list, n_participants_list, level_mapping=level_mapping)
        
        # Calculate FMI for all terms
        fmi_stats = {}
        if ests_reml:
            for p in ests_reml[0].keys():
                ests_p = [d[p] for d in ests_reml if p in d]
                ses_p = [d[p] for d in bse_reml if p in d]
                if len(ests_p) < len(ests_reml) or len(ests_p) == 0: continue
                W = np.mean([se**2 for se in ses_p])
                B = np.var(ests_p, ddof=1) if len(ests_reml) > 1 else 0
                if W > 0:
                    r = (1 + 1/len(ests_reml)) * (B / W)
                    fmi = r / (1 + r)
                    fmi_stats[p] = {"FMI": round(fmi, 4), "B_var": round(B, 6), "W_var": round(W, 6)}
        
        is_singular = bool(np.mean(iccs_reml) < 1e-4) if iccs_reml else False
        rs_improves = bool(np.mean(rs_stats["aic"]) < np.mean(ml_stats["aic"])) if rs_stats["aic"] and ml_stats["aic"] else None
        
        all_metrics[model_name] = {
            "Converged": all(reml_stats["conv"]),
            "Converged_ML": all(ml_stats["conv"]),
            "Dataset": dataset_name,
            "N_obs_mean": np.mean(n_obs_list),
            "m_imputations": len(ests_ml),
            "ICC_mean_ML": np.mean(iccs_ml) if iccs_ml else None,
            "ICC_mean_REML": np.mean(iccs_reml) if iccs_reml else None,
            "Pseudo_R2_Marginal_mean": np.mean(ml_stats["r2_m"]) if ml_stats["r2_m"] else None,
            "Pseudo_R2_Conditional_mean": np.mean(ml_stats["r2_c"]) if ml_stats["r2_c"] else None,
            "Singular_Fit_Warning": is_singular,
            "Random_Slope_Improves_Fit": rs_improves,
            "AIC_ML": {
                "mean": np.mean(ml_stats["aic"]),
                "sd": np.std(ml_stats["aic"]) if len(ml_stats["aic"]) > 1 else 0.0,
                "min": np.min(ml_stats["aic"]),
                "max": np.max(ml_stats["aic"]),
            },
            "BIC_ML": {
                "mean": np.mean(ml_stats["bic"]),
                "sd": np.std(ml_stats["bic"]) if len(ml_stats["bic"]) > 1 else 0.0,
                "min": np.min(ml_stats["bic"]),
                "max": np.max(ml_stats["bic"]),
            },
            "LogLikelihood_ML": {
                "mean": np.mean(ml_stats["llf"]),
                "sd": np.std(ml_stats["llf"]) if len(ml_stats["llf"]) > 1 else 0.0,
                "min": np.min(ml_stats["llf"]),
                "max": np.max(ml_stats["llf"]),
            },
            "Note": "AIC/BIC/LogLikelihood are averaged across imputations (descriptive) and are not formally pooled. Coefficients/SEs are formally pooled.",
            "Parameters": pooled_reml, # REML is primary for presentation
            "Parameters_ML": pooled_ml,
            "FMI_Statistics": fmi_stats
        }
        
        print(f"  [{model_name}] Converged successfully. Pooled m={len(ests_ml)} datasets.")
        
        summary_txt = f"\n{'='*60}\n  {model_name} (REML fit for estimates, ML for comparison)\n{'='*60}\n"
        summary_txt += f"Formula: {nonlinear_formula}\n"
        summary_txt += f"N (mean) = {np.mean(n_obs_list):.1f}, Imputations = {len(ests_ml)}\n"
        summary_txt += f"AIC (ML mean) = {np.mean(ml_stats['aic']):.4f}, BIC (ML mean) = {np.mean(ml_stats['bic']):.4f}\n"
        if iccs_reml: summary_txt += f"ICC (REML mean) = {np.mean(iccs_reml):.4f}\n\n"
        
        df_res = pd.DataFrame(pooled_reml).T
        summary_txt += df_res.to_string() + "\n"
        
        if 'Workload_Baseline_WP' in pooled_reml and 'Workload_Crunch_WP' in pooled_reml:
            baseline_coef = pooled_reml['Workload_Baseline_WP']['Estimate']
            crunch_coef = pooled_reml['Workload_Crunch_WP']['Estimate']
            ratio = crunch_coef / baseline_coef if baseline_coef != 0 else float('inf')
            summary_txt += f"\n  Asymmetry Check:\n"
            summary_txt += f"    Baseline slope (normal days): {baseline_coef:.4f}\n"
            summary_txt += f"    Crunch slope (tipping point): {crunch_coef:.4f}\n"
            summary_txt += f"    Ratio (Crunch / Baseline):    {ratio:.2f}x multiplier\n"
            
        all_summaries.append(summary_txt)
        
    with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(all_summaries))
    with open(METRICS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, default=str)
    print("\n[+] Stage 4 Complete. Proceed to Stage 5.")

if __name__ == "__main__":
    run_frequentist_nonlinear_models()
