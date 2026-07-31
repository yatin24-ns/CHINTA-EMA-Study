from stats_utils import BASE_DIR, DATASETS, apply_rubins_rules 
import os
import scipy.stats
import warnings
import json
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SUMMARY_OUTPUT = os.path.join(BASE_DIR, "frequentist_linear_summary.txt")
METRICS_OUTPUT = os.path.join(BASE_DIR, "frequentist_linear_metrics.json")

def prepare_data(df):
    required_cols = [
        'S2_Fatigue_Mean_z', 'Workload_WP_z', 'Leisure_WP_z', 
        'Sleep_WP_z',
        'Workload_BP_z', 'Leisure_BP_z', 'Sleep_BP_z',
        'Trait_Fatigue_z', 'day_number_z', 'is_weekend',
        'Participant_No'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  [WARNING] Dropping dataset: Missing required columns: {missing}")
        return None
    return df.dropna(subset=['S2_Fatigue_Mean_z']).copy()

def run_frequentist_linear_models():
    print("=" * 70)
    print("  PIPELINE STAGE 2: FREQUENTIST LINEAR MODELS")
    print("=" * 70)
    # Formula: interaction and week-boundary controls removed based on LRT
    # (LRT p>0.50 for interaction and p>0.38 for onset/finish across all 20 imputations)
    linear_formula = (
        "S2_Fatigue_Mean_z ~ "
        "Workload_WP_z + Leisure_WP_z + "
        "Sleep_WP_z + "
        "Workload_BP_z + Leisure_BP_z + Sleep_BP_z + Trait_Fatigue_z + "
        "day_number_z + is_weekend"
    )
    # Full formula retained for LRT comparison (dropped terms diagnostic)
    linear_formula_full = (
        "S2_Fatigue_Mean_z ~ "
        "Workload_WP_z + Leisure_WP_z + Workload_WP_z:Leisure_WP_z + "
        "Sleep_WP_z + "
        "Workload_BP_z + Leisure_BP_z + Sleep_BP_z + Trait_Fatigue_z + "
        "day_number_z + is_week_onset + is_week_finish + is_weekend"
    )
    
    all_summaries = []
    all_metrics = {}
    
    # Run on original and PASSIVE imputed datasets (passive is correct for interactions)
    datasets_to_check = {
        "original": DATASETS["original"],
        "imputed": DATASETS["imputed_passive"]
    }
    
    for dataset_name, dataset_path in datasets_to_check.items():
        if not os.path.exists(dataset_path):
            continue
            
        print(f"\n\n  DATASET: {dataset_name.upper()}\n{'-' * 70}")
        
        if dataset_name == "original":
            df = prepare_data(pd.read_excel(dataset_path))
            datasets_to_run = [(1, df)]
        else:
            df_full = pd.read_parquet(dataset_path)
            datasets_to_run = []
            for imp in sorted(df_full['.imp'].unique()):
                datasets_to_run.append((imp, prepare_data(df_full[df_full['.imp'] == imp])))
                
        # Fit only 2L Linear Model
        model_name = f"Linear_2L_{dataset_name}"
        print(f"\n  [{model_name}] Fitting...")
        
        ml_stats = {"aic": [], "bic": [], "llf": [], "conv": []}
        reml_stats = {"conv": []}
        
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
                model_ml = smf.mixedlm(linear_formula, df_imp, groups=df_imp["Participant_No"])
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
                    icc_ml = float(res_ml.cov_re.iloc[0, 0]) / (float(res_ml.cov_re.iloc[0, 0]) + float(res_ml.scale))
                    iccs_ml.append(icc_ml)
                    
                # 2. Fit REML model
                model_reml = smf.mixedlm(linear_formula, df_imp, groups=df_imp["Participant_No"])
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
                failed = True; break
                
        if failed or not ests_ml:
            all_metrics[model_name] = {"Converged": False, "Dataset": dataset_name}
            print(f"  [WARNING] Estimation failed for {model_name}. Skipping pooling.")
            continue
            
        pooled_ml = apply_rubins_rules(ests_ml, bse_ml, n_obs_list, n_participants_list)
        pooled_reml = apply_rubins_rules(ests_reml, bse_reml, n_obs_list, n_participants_list)
        
        # Calculate FMI for the interaction term
        fmi_stats = {}
        for p in ["Workload_WP_z:Leisure_WP_z"]:
            if p in ests_reml[0]:
                ests_p = [d[p] for d in ests_reml]
                ses_p = [d[p] for d in bse_reml]
                W = np.mean([se**2 for se in ses_p])
                B = np.var(ests_p, ddof=1) if len(ests_reml) > 1 else 0
                if W > 0:
                    r = (1 + 1/len(ests_reml)) * (B / W)
                    fmi = r / (1 + r)
                    fmi_stats[p] = {"FMI": round(fmi, 4), "B_var": round(B, 6), "W_var": round(W, 6)}
                    
        all_metrics[model_name] = {
            "Converged": all(reml_stats["conv"]),
            "Converged_ML": all(ml_stats["conv"]),
            "Dataset": dataset_name,
            "N_obs_mean": np.mean(n_obs_list),
            "m_imputations": len(ests_ml),
            "ICC_mean_ML": np.mean(iccs_ml) if iccs_ml else None,
            "ICC_mean_REML": np.mean(iccs_reml) if iccs_reml else None,
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
            "Parameters": pooled_reml,
            "Parameters_ML": pooled_ml,
            "FMI_Statistics": fmi_stats,
            "Per_Imputation_Convergence_REML": reml_stats["conv"],
            "Per_Imputation_Convergence_ML": ml_stats["conv"],
        }
        
        print(f"  [{model_name}] Converged successfully. Pooled m={len(ests_ml)} datasets.")
        
        # ---- Inline LRT diagnostics (only for imputed, multi-imputation datasets) ----
        if dataset_name != "original" and len(datasets_to_run) > 1:
            print(f"\n  {'-'*60}")
            print(f"  INLINE LRT -- {model_name}")
            print(f"  {'-'*60}")
            from scipy.stats import chi2 as chi2_dist
            lrt_tests = [
                ("Interaction (Workload_WP_z:Leisure_WP_z)",
                 linear_formula + " + Workload_WP_z:Leisure_WP_z",
                 linear_formula, 1),
                ("is_week_onset + is_week_finish (joint)",
                 linear_formula + " + is_week_onset + is_week_finish",
                 linear_formula, 2),

            ]
            for lrt_label, f_full, f_red, df_lrt in lrt_tests:
                lrs, ps = [], []
                for imp, df_imp in datasets_to_run:
                    if df_imp is None or len(df_imp) == 0:
                        continue
                    try:
                        try:
                            r_full = smf.mixedlm(f_full, df_imp, groups=df_imp["Participant_No"]).fit(reml=False, method='lbfgs')
                        except Exception:
                            r_full = smf.mixedlm(f_full, df_imp, groups=df_imp["Participant_No"]).fit(reml=False, method='bfgs')
                            
                        try:
                            r_red  = smf.mixedlm(f_red,  df_imp, groups=df_imp["Participant_No"]).fit(reml=False, method='lbfgs')
                        except Exception:
                            r_red  = smf.mixedlm(f_red,  df_imp, groups=df_imp["Participant_No"]).fit(reml=False, method='bfgs')
                            
                        lr = 2 * (r_full.llf - r_red.llf)
                        lrs.append(lr)
                        ps.append(chi2_dist.sf(lr, df=df_lrt))
                    except Exception as e:
                        print(f"    [WARN] LRT fit failed on imp {imp}: {e}")
                if lrs:
                    mean_p = np.mean(ps)
                    n_sig = sum(p < 0.05 for p in ps)
                    verdict = "DROP " if mean_p > 0.10 else ("KEEP (justified)" if mean_p < 0.05 else "BORDERLINE")
                    print(f"  LRT: {lrt_label}")
                    print(f"       mean LR={np.mean(lrs):.4f}  mean p={mean_p:.4f}  sig={n_sig}/{len(lrs)}  -> {verdict}")
            print(f"  {'-'*60}\n")
        

        # Create a summary text
        summary_txt = f"\n{'='*60}\n  {model_name} (REML fit for estimates, ML for comparison)\n{'='*60}\n"
        summary_txt += f"Formula: {linear_formula}\n"
        summary_txt += f"N (mean) = {np.mean(n_obs_list):.1f}, Imputations = {len(ests_ml)}\n"
        summary_txt += f"AIC (ML mean) = {np.mean(ml_stats['aic']):.4f}, BIC (ML mean) = {np.mean(ml_stats['bic']):.4f}\n"
        if iccs_reml: summary_txt += f"ICC (REML mean) = {np.mean(iccs_reml):.4f}\n\n"
        
        df_res = pd.DataFrame(pooled_reml).T
        summary_txt += df_res.to_string() + "\n"
        all_summaries.append(summary_txt)
            
    with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(all_summaries))
    with open(METRICS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, default=str)
    print("\n[+] Stage 2 Complete. Proceed to Stage 3.")

if __name__ == "__main__":
    run_frequentist_linear_models()
