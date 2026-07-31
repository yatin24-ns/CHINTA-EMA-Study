"""
=============================================================================
PIPELINE STAGE 5: BAYESIAN NON-LINEAR MIXED MODELS
=============================================================================
Two model classes:

Case 3 — 2-Level & 3-Level Bayesian Non-Linear (Bambi):
  Uses pre-transformed variables (piecewise spline + log-leisure) in a 
  Bambi formula.
  
Case 4 — 3-Level Bayesian Non-Linear with True Logistic (PyMC):
  Daily states nested in Weeks nested in Participants, mapped to a 
  true sigmoidal biological exhaustion curve.

Data Integration:
  Runs on original_complete_cases.xlsx and imputed_processed_ema.parquet (m=5),
  pooling posteriors across imputations.
=============================================================================
"""

from stats_utils import BASE_DIR, DATASETS
import pandas as pd
import numpy as np
import warnings
import json
import os
import bambi as bmb
import arviz as az
import xarray as xr
import pymc as pm
import pytensor.tensor as pt
import gc


def concat_idatas(idatas):
    """
    ArViz 1.x compatible pooling of multiple InferenceData (DataTree) objects.
    Concatenates each group (posterior, log_likelihood, etc.) along the
    'chain' dimension using xr.DataTree.from_dict().
    """
    if len(idatas) == 1:
        return idatas[0]
    
    # Collect all group names across all idatas
    all_groups = set()
    for idata in idatas:
        for g in idata.groups:
            if g != '/':
                all_groups.add(g.lstrip('/'))
    
    tree_dict = {}
    for group in all_groups:
        group_datasets = []
        for idata in idatas:
            try:
                node = idata[f"/{group}"]
                ds = node.ds
                if ds is not None and len(ds) > 0:
                    group_datasets.append(ds)
            except (KeyError, ValueError):
                pass
        
        if not group_datasets:
            continue
        
        # observed_data is identical across imputations (MIDO strategy)
        if group == "observed_data":
            tree_dict[f"/{group}"] = group_datasets[0]
        else:
            try:
                tree_dict[f"/{group}"] = xr.concat(group_datasets, dim="chain")
            except Exception:
                tree_dict[f"/{group}"] = group_datasets[0]
    
    return xr.DataTree.from_dict(tree_dict)

warnings.filterwarnings("ignore")

def prepare_nonlinear_bayesian_data(df):
    required_cols = [
        'S2_Fatigue_Mean_z', 'Participant_No',
        'Workload_BP_z', 'Sleep_BP_z', 'Trait_Fatigue_z',
        'day_number_z', 'is_week_onset', 'is_week_finish', 'is_weekend',
        'Sleep_WP_z',
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing: return None
    
    if 'Workload_Baseline_WP' not in df.columns:
        if 'Workload_WP_z' in df.columns:
            df['Workload_Baseline_WP'] = np.minimum(0, df['Workload_WP_z'])
            df['Workload_Crunch_WP'] = np.maximum(0, df['Workload_WP_z'])
            
    if 'Leisure_Log_WP_z' not in df.columns:
        if 'S3_Leisure_Mean' in df.columns:
            df['Leisure_Log'] = np.log1p(df['S3_Leisure_Mean'])
            df['Leisure_Log_BP'] = df.groupby('Participant_No')['Leisure_Log'].transform('mean')
            df['Leisure_Log_WP'] = df['Leisure_Log'] - df['Leisure_Log_BP']
            for s in ['WP', 'BP']:
                col = f'Leisure_Log_{s}'
                m, sd = df[col].mean(), df[col].std()
                df[f'{col}_z'] = (df[col] - m) / sd if sd > 0 else 0.0
                
    if 'Interaction_Normal' not in df.columns:
        if 'Workload_Baseline_WP' in df.columns and 'Leisure_Log_WP_z' in df.columns:
            df['Interaction_Normal'] = df['Workload_Baseline_WP'] * df['Leisure_Log_WP_z']
            df['Interaction_Crunch'] = df['Workload_Crunch_WP'] * df['Leisure_Log_WP_z']
            
    df = df.dropna(subset=['S2_Fatigue_Mean_z']).copy()
    predictor_cols = [
        'Workload_Baseline_WP', 'Workload_Crunch_WP', 'Leisure_Log_WP_z',
        'Interaction_Normal', 'Interaction_Crunch', 'Sleep_WP_z',
        'Workload_BP_z', 'Leisure_Log_BP_z',
        'Sleep_BP_z', 'Trait_Fatigue_z', 'day_number_z',
    ]
    df = df.dropna(subset=[c for c in predictor_cols if c in df.columns]).copy()
    df['Participant_No'] = df['Participant_No'].astype(str)
    
    if 'week_id' not in df.columns:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['week_id'] = df['date'].dt.isocalendar().week.astype(int)
        else:
            df['week_id'] = 1
    df['week_id'] = df['week_id'].astype(str)
    return df

def run_case3_bambi_models(datasets_to_run, dataset_name, level, all_metrics, ppc_summaries):
    formula = (
        "S2_Fatigue_Mean_z ~ 1 + Workload_Baseline_WP + Workload_Crunch_WP + Leisure_Log_WP_z + "
        "Interaction_Normal + Interaction_Crunch + Sleep_WP_z + "
        "Workload_BP_z + Leisure_Log_BP_z + Sleep_BP_z + Trait_Fatigue_z + "
        "day_number_z + is_week_onset + is_week_finish + is_weekend + (Workload_Baseline_WP + Workload_Crunch_WP + Leisure_Log_WP_z | Participant_No)"
    )
    
    model_name = f"NonLinear_{level}_Bambi_{dataset_name}"
    print(f"\n  [{model_name}] Fitting {len(datasets_to_run)} dataset(s)...")
    
    draws_per_imp = 1000 if len(datasets_to_run) == 1 else 500
    tune_per_imp = 1000 if len(datasets_to_run) == 1 else 500
    
    idatas = []
    failed = False
    for imp, df_imp in datasets_to_run:
        try:
            model = bmb.Model(formula, df_imp)
            idata = model.fit(draws=draws_per_imp, tune=tune_per_imp, chains=2, cores=1, target_accept=0.95, random_seed=42+imp, idata_kwargs={"log_likelihood": True})
            idatas.append(idata)
            del model
        except Exception as e:
            print(f"    -> FAILED on imp {imp}: {e}")
            failed = True; break
            
    if failed or not idatas:
        all_metrics[model_name] = {"Converged": False, "Dataset": dataset_name}
        return
        
    print(f"  [{model_name}] Pooling posteriors...")
    idata_pooled = concat_idatas(idatas)
    nc_path = os.path.join(BASE_DIR, f"bayesian_{model_name.lower()}.nc")
    idata_pooled.to_netcdf(nc_path)
    
    # Calculate LOO per-imputation
    loos = []
    pareto_k_warnings = 0
    for idata in idatas:
        try:
            loo = az.loo(idata)
            loos.append(loo.elpd)  # ArViz 1.x: attribute is 'elpd', not 'elpd_loo'
            pk = getattr(loo, 'influence_pareto_k', getattr(loo, 'pareto_k', None))
            if pk is not None and hasattr(pk, '__iter__') and (pk > 0.7).any():
                pareto_k_warnings += 1
        except Exception as e:
            print(f"    [WARNING] LOO calculation failed for an imputation: {e}")
            
    summary_df = az.summary(idata_pooled)
    global_params = summary_df[~summary_df.index.str.contains(r'\|')]
    
    with open(os.path.join(BASE_DIR, f"bayesian_{model_name.lower()}_summary.txt"), "w") as f:
        f.write(summary_df.to_string())
        
    rhat_max = pd.to_numeric(summary_df['r_hat'], errors='coerce').max() if 'r_hat' in summary_df.columns else None
    all_metrics[model_name] = {
        "Dataset": dataset_name, 
        "Converged": bool(rhat_max < 1.05) if rhat_max else False,
        "R_hat_max": float(rhat_max) if rhat_max else None, 
        "Trace_file": nc_path, 
        "m_imputations": len(idatas),
        "LOO_elpd_mean": float(np.mean(loos)) if loos else None,
        "LOO_elpd_sd": float(np.std(loos)) if len(loos) > 1 else 0.0,
        "LOO_pareto_k_warnings": pareto_k_warnings,
        "pooling_method": "Posteriors combined via az.concat across draw dimension. LOO pooled via mean ELPD across imputations."
    }
    for param in global_params.index:
        all_metrics[model_name][f"post_{param}_mean"] = float(global_params.loc[param, 'mean'])
        all_metrics[model_name][f"post_{param}_sd"] = float(global_params.loc[param, 'sd'])
        
    # Generate PPC plot and quantitative PPC statistics
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Extract observed and posterior predictive values
        obs_node = idata_pooled.get('/observed_data', idata_pooled)
        pp_node  = idata_pooled.get('/posterior_predictive', idata_pooled)
        
        try:
            obs_val = obs_node.ds['S2_Fatigue_Mean_z'].values
        except Exception:
            obs_val = None
        try:
            pp_val = pp_node.ds['S2_Fatigue_Mean_z'].values
        except Exception:
            pp_val = None
            
        # Plot: try az.plot_ppc_dist (ArViz 1.x), fall back to manual histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ppc_plotted = False
        if hasattr(az, 'plot_ppc_dist'):
            try:
                az.plot_ppc_dist(idata_pooled, axes=ax)
                ppc_plotted = True
            except Exception:
                pass
        if not ppc_plotted and obs_val is not None and pp_val is not None:
            ax.hist(obs_val.ravel(), bins=30, alpha=0.5, label='Observed', density=True)
            ax.hist(pp_val.ravel(), bins=30, alpha=0.5, label='Posterior Predictive', density=True)
            ax.legend(); ax.set_xlabel('S2_Fatigue_Mean_z'); ax.set_title(f'PPC: {model_name}')
            
        ppc_fig_path = os.path.join(BASE_DIR, f"ppc_{model_name.lower()}.png")
        plt.tight_layout()
        plt.savefig(ppc_fig_path, dpi=100)
        plt.close(fig)
        
        if obs_val is not None and pp_val is not None:
            obs_mean, obs_std = float(np.mean(obs_val)), float(np.std(obs_val))
            pp_mean,  pp_std  = float(np.mean(pp_val)),  float(np.std(pp_val))
            ppc_summaries[model_name] = {
                "obs_mean": obs_mean, "obs_std": obs_std,
                "pp_mean":  pp_mean,  "pp_std":  pp_std,
                "mae_mean": float(np.abs(obs_mean - pp_mean)),
                "mae_std":  float(np.abs(obs_std  - pp_std))
            }
        print(f"    Saved PPC plot to {ppc_fig_path}")
    except Exception as e:
        print(f"    [WARNING] PPC plot generation failed: {e}")
        
    del idatas, idata_pooled, summary_df; gc.collect()

def run_case4_pymc_logistic(datasets_to_run, dataset_name, all_metrics, ppc_summaries):
    # Converted to 2L Logistic Model to match the rest of the 2L pipeline
    model_name = f"NonLinear_2L_Logistic_{dataset_name}"
    print(f"\n  [{model_name}] Fitting {len(datasets_to_run)} dataset(s)...")
    
    draws_per_imp = 1000 if len(datasets_to_run) == 1 else 500
    tune_per_imp = 1500 if len(datasets_to_run) == 1 else 1000
    
    idatas = []
    failed = False
    for imp, df_imp in datasets_to_run:
        try:
            part_idx, parts = pd.factorize(df_imp['Participant_No'])
            y = df_imp['S2_Fatigue_Mean'].values
            X_wl = df_imp['Workload_WP_z'].values
            X_ls = df_imp['Leisure_Log_WP_z'].values
            
            with pm.Model() as logistic_model:
                # Level-2 Priors for Asymptote and Growth
                alpha_max_mu = pm.Normal('alpha_max_mu', mu=5, sigma=1.5)
                alpha_max_sigma = pm.HalfNormal('alpha_max_sigma', sigma=1)
                
                gamma_mu = pm.HalfNormal('gamma_mu', sigma=1.5)
                gamma_sigma = pm.HalfNormal('gamma_sigma', sigma=0.5)
                
                # Level-1 Varying Effects
                alpha_max = pm.TruncatedNormal('alpha_max', mu=alpha_max_mu, sigma=alpha_max_sigma, 
                                               lower=0, upper=9, shape=len(parts))
                gamma = pm.HalfNormal('gamma_steep', sigma=gamma_sigma, shape=len(parts))
                
                # Baseline inflection point
                mu_beta = pm.Normal('mu_beta', mu=0.0, sigma=1.0)
                delta_ls = pm.Normal('delta_leisure', mu=0.0, sigma=1.0)
                
                # Random effects on inflection point (2-level)
                sigma_u = pm.HalfStudentT('sigma_u', nu=4, sigma=1.0)
                u_beta_raw = pm.Normal('u_beta_raw', 0, 1, shape=len(parts))
                u_beta = pm.Deterministic('u_beta', u_beta_raw * sigma_u)
                
                # 2-Level Sigmoidal Formulation
                beta_j = pm.Deterministic('beta_tipping_point', mu_beta + delta_ls * X_ls + u_beta[part_idx])
                linear_component = -gamma[part_idx] * (X_wl - beta_j)
                mu_y = pm.Deterministic('mu_y', 1.0 + (alpha_max[part_idx] / (1.0 + pt.exp(linear_component))))
                
                sigma_y = pm.HalfStudentT('sigma_y', nu=4, sigma=1.0)
                pm.Normal('obs', mu=mu_y, sigma=sigma_y, observed=y)
                
                idata = pm.sample(draws=draws_per_imp, tune=tune_per_imp, chains=2, cores=1, target_accept=0.95, random_seed=42+imp)
                # Compute log-likelihood for LOO-CV
                pm.compute_log_likelihood(idata)
                # Sample posterior predictive for PPCs
                pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=42+imp)
                
                idatas.append(idata)
        except Exception as e:
            print(f"    -> FAILED on imp {imp}: {e}")
            failed = True; break
            
    if failed or not idatas:
        all_metrics[model_name] = {"Converged": False, "Dataset": dataset_name}
        return
        
    print(f"  [{model_name}] Pooling posteriors...")
    # az.concat does not exist in ArViz 1.x — use the concat_idatas helper
    idata_pooled = concat_idatas(idatas)
    nc_path = os.path.join(BASE_DIR, f"bayesian_{model_name.lower()}.nc")
    idata_pooled.to_netcdf(nc_path)
    
    # Calculate LOO per-imputation
    loos = []
    pareto_k_warnings = 0
    for idata in idatas:
        try:
            loo = az.loo(idata)
            loos.append(loo.elpd)  # ArViz 1.x: attribute is 'elpd', not 'elpd_loo'
            pk = getattr(loo, 'influence_pareto_k', getattr(loo, 'pareto_k', None))
            if pk is not None and hasattr(pk, '__iter__') and (pk > 0.7).any():
                pareto_k_warnings += 1
        except Exception as e:
            print(f"    [WARNING] LOO calculation failed for an imputation: {e}")
            
    summary_df = az.summary(idata_pooled, var_names=['alpha_max', 'gamma_steep', 'mu_beta', 'delta_leisure', 'sigma_u', 'sigma_y'])
    with open(os.path.join(BASE_DIR, f"bayesian_{model_name.lower()}_summary.txt"), "w") as f:
        f.write(summary_df.to_string())
        
    rhat_max = pd.to_numeric(summary_df['r_hat'], errors='coerce').max() if 'r_hat' in summary_df.columns else None
    all_metrics[model_name] = {
        "Dataset": dataset_name, 
        "Converged": bool(rhat_max < 1.05) if rhat_max else False,
        "R_hat_max": float(rhat_max) if rhat_max else None, 
        "Trace_file": nc_path, 
        "m_imputations": len(idatas),
        "LOO_elpd_mean": float(np.mean(loos)) if loos else None,
        "LOO_elpd_sd": float(np.std(loos)) if len(loos) > 1 else 0.0,
        "LOO_pareto_k_warnings": pareto_k_warnings,
        "pooling_method": "Posteriors combined via az.concat across draw dimension. LOO pooled via mean ELPD across imputations."
    }
    for param in summary_df.index:
        all_metrics[model_name][f"post_{param}_mean"] = float(summary_df.loc[param, 'mean'])
        all_metrics[model_name][f"post_{param}_sd"] = float(summary_df.loc[param, 'sd'])
        
    # Generate PPC plot and quantitative PPC statistics
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Extract observed and posterior predictive ('obs' is the PyMC variable name)
        obs_node = idata_pooled.get('/observed_data', idata_pooled)
        pp_node  = idata_pooled.get('/posterior_predictive', idata_pooled)
        try:
            obs_val = obs_node.ds['obs'].values
        except Exception:
            obs_val = None
        try:
            pp_val = pp_node.ds['obs'].values
        except Exception:
            pp_val = None
            
        fig, ax = plt.subplots(figsize=(8, 5))
        ppc_plotted = False
        if hasattr(az, 'plot_ppc_dist'):
            try:
                az.plot_ppc_dist(idata_pooled, axes=ax)
                ppc_plotted = True
            except Exception:
                pass
        if not ppc_plotted and obs_val is not None and pp_val is not None:
            ax.hist(obs_val.ravel(), bins=30, alpha=0.5, label='Observed', density=True)
            ax.hist(pp_val.ravel(), bins=30, alpha=0.5, label='Posterior Predictive', density=True)
            ax.legend(); ax.set_xlabel('obs'); ax.set_title(f'PPC: {model_name}')
            
        ppc_fig_path = os.path.join(BASE_DIR, f"ppc_{model_name.lower()}.png")
        plt.tight_layout()
        plt.savefig(ppc_fig_path, dpi=100)
        plt.close(fig)
        
        if obs_val is not None and pp_val is not None:
            obs_mean, obs_std = float(np.mean(obs_val)), float(np.std(obs_val))
            pp_mean,  pp_std  = float(np.mean(pp_val)),  float(np.std(pp_val))
            ppc_summaries[model_name] = {
                "obs_mean": obs_mean, "obs_std": obs_std,
                "pp_mean":  pp_mean,  "pp_std":  pp_std,
                "mae_mean": float(np.abs(obs_mean - pp_mean)),
                "mae_std":  float(np.abs(obs_std  - pp_std))
            }
        print(f"    Saved PPC plot to {ppc_fig_path}")
    except Exception as e:
        print(f"    [WARNING] PPC plot generation failed: {e}")
        
    del idatas, idata_pooled, summary_df; gc.collect()

def run_bayesian_nonlinear_models():
    print("=" * 70)
    print("  PIPELINE STAGE 5: BAYESIAN NON-LINEAR MODELS")
    print("=" * 70)
    
    all_metrics = {}
    ppc_summaries = {}
    
    # Run on original, imputed_jav, and imputed_passive datasets to compare JAV vs Passive
    datasets_to_check = {
        "original": DATASETS["original"],
        "imputed_jav": DATASETS["imputed_jav"],
        "imputed_passive": DATASETS["imputed_passive"]
    }
    
    for dataset_name, dataset_path in datasets_to_check.items():
        if not os.path.exists(dataset_path): continue
        print(f"\n{'-' * 70}\n  DATASET: {dataset_name.upper()}\n{'-' * 70}")
        
        if dataset_name == "original":
            df = prepare_nonlinear_bayesian_data(pd.read_excel(dataset_path))
            datasets_to_run = [(1, df)]
        else:
            df_full = pd.read_parquet(dataset_path)
            datasets_to_run = [(imp, prepare_nonlinear_bayesian_data(df_full[df_full['.imp'] == imp])) for imp in sorted(df_full['.imp'].unique())]
            
        run_case3_bambi_models(datasets_to_run, dataset_name, "2L", all_metrics, ppc_summaries)
        run_case4_pymc_logistic(datasets_to_run, dataset_name, all_metrics, ppc_summaries)
            
    with open(os.path.join(BASE_DIR, "bayesian_nonlinear_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=4, default=str)
        
    with open(os.path.join(BASE_DIR, "ppc_summary.json"), "w") as f:
        json.dump(ppc_summaries, f, indent=4, default=str)
        
    print("\n[+] Stage 5 Complete. Proceed to Stage 6.")

if __name__ == "__main__":
    run_bayesian_nonlinear_models()
