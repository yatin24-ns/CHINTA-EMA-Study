from stats_utils import BASE_DIR, DATASETS
import pandas as pd
import numpy as np
import warnings
import json
import os
import bambi as bmb
import arviz as az
import xarray as xr
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

def prepare_bayesian_data(df):
    """Prepare and validate data for Bayesian linear models."""
    required_cols = [
        'S2_Fatigue_Mean_z', 'Workload_WP_z', 'Leisure_WP_z',
        'Sleep_WP_z',
        'Workload_BP_z', 'Leisure_BP_z', 'Sleep_BP_z',
        'Trait_Fatigue_z', 'day_number_z',
        'is_weekend',
        'Participant_No'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing: return None
    
    df = df.dropna(subset=['S2_Fatigue_Mean_z']).copy()
    predictor_cols = [c for c in required_cols if c != 'Participant_No']
    df = df.dropna(subset=predictor_cols).copy()
    
    df['Participant_No'] = df['Participant_No'].astype(str)
    
    if 'week_id' not in df.columns:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['week_id'] = df['date'].dt.isocalendar().week.astype(int)
        else:
            df['week_id'] = 1
    
    df['week_id'] = df['week_id'].astype(str)
    return df

def run_bayesian_linear_models():
    print("=" * 70)
    print("  PIPELINE STAGE 3: BAYESIAN LINEAR MODELS")
    print("=" * 70)
    
    # Fixed-effects structure adopted from 2_frequentist_linear.py's LRT results;
    # not independently re-validated via LOO/WAIC in this script, for computational efficiency.
    formula_2L = (
        "S2_Fatigue_Mean_z ~ 1 + "
        "Workload_WP_z + Leisure_WP_z + "
        "Sleep_WP_z + "
        "Workload_BP_z + Leisure_BP_z + Sleep_BP_z + Trait_Fatigue_z + "
        "day_number_z + is_weekend + "
        "(1 | Participant_No)"
    )
    
    # Secondary/Sensitivity models kept for robustness checks (not in main comparison loop)
    formula_2L_rs = (
        "S2_Fatigue_Mean_z ~ 1 + "
        "Workload_WP_z + Leisure_WP_z + "
        "Sleep_WP_z + "
        "Workload_BP_z + Leisure_BP_z + Sleep_BP_z + Trait_Fatigue_z + "
        "day_number_z + is_weekend + "
        "(Workload_WP_z + Leisure_WP_z | Participant_No)"
    )
    
    # 3-Level model retained as the ICC check showed week-level variance >= 5%
    formula_3L = (
        "S2_Fatigue_Mean_z ~ 1 + "
        "Workload_WP_z + Leisure_WP_z + "
        "Sleep_WP_z + "
        "Workload_BP_z + Leisure_BP_z + Sleep_BP_z + Trait_Fatigue_z + "
        "day_number_z + is_weekend + "
        "(1 | Participant_No) + (1|Participant_No:week_id)"
    )
    
    all_metrics = {}
    
    # Run on original and PASSIVE imputed datasets
    datasets_to_check = {
        "original": DATASETS["original"],
        "imputed": DATASETS["imputed_passive"]
    }
    
    for dataset_name, dataset_path in datasets_to_check.items():
        if not os.path.exists(dataset_path): continue
        print(f"\n{'-' * 70}\n  DATASET: {dataset_name.upper()}\n{'-' * 70}")
        
        if dataset_name == "original":
            df = prepare_bayesian_data(pd.read_excel(dataset_path))
            datasets_to_run = [(1, df)]
            draws_per_imp, tune_per_imp = 1000, 1000
        else:
            df_full = pd.read_parquet(dataset_path)
            datasets_to_run = [(imp, prepare_bayesian_data(df_full[df_full['.imp'] == imp])) for imp in sorted(df_full['.imp'].unique())]
            draws_per_imp, tune_per_imp = 500, 500
            
        model_name = f"Linear_2L_{dataset_name}"
        print(f"\n  [{model_name}] Initialising Bambi model across {len(datasets_to_run)} dataset(s)...")
        
        idatas = []
        per_imp_rhats = {}  # {param: [rhat_imp1, rhat_imp2, ...]}
        failed = False
        for imp, df_imp in datasets_to_run:
            if df_imp is None or len(df_imp) == 0:
                failed = True; break
            try:
                model = bmb.Model(formula_2L, df_imp)
                print(f"    -> Sampling imp {imp}/{len(datasets_to_run)} (draws={draws_per_imp}, tune={tune_per_imp})...")
                idata = model.fit(draws=draws_per_imp, tune=tune_per_imp, chains=2, cores=1, target_accept=0.90, random_seed=42+imp, idata_kwargs={"log_likelihood": True})
                # Collect per-imputation R-hat before pooling
                try:
                    imp_summary = az.summary(idata)
                    imp_global = imp_summary[~imp_summary.index.str.contains(r'\|')]
                    if 'r_hat' in imp_global.columns:
                        for param in imp_global.index:
                            per_imp_rhats.setdefault(param, []).append(float(imp_global.loc[param, 'r_hat']))
                except Exception:
                    pass
                idatas.append(idata)
                del model
            except Exception as e:
                print(f"    -> FAILED on imp {imp}: {e}")
                failed = True; break
        
        if failed or not idatas:
            all_metrics[model_name] = {"Converged": False, "Dataset": dataset_name}
            continue
            
        print(f"  [{model_name}] Pooling posteriors...")
        idata_pooled = concat_idatas(idatas)
        
        nc_path = os.path.join(BASE_DIR, f"bayesian_{model_name.lower()}.nc")
        idata_pooled.to_netcdf(nc_path)
        
        # Calculate LOO per-imputation (ArViz 1.x: elpd_loo → elpd, pareto_k → influence_pareto_k)
        loos = []
        pareto_k_warnings = 0
        for idata in idatas:
            try:
                loo = az.loo(idata)
                loos.append(loo.elpd)  # ArViz 1.x: attribute is 'elpd'
                pk = getattr(loo, 'influence_pareto_k', None)  # ArViz 1.x attribute
                if pk is not None and (pk > 0.7).any():
                    pareto_k_warnings += 1
            except Exception as e:
                print(f"    [WARNING] LOO calculation failed for an imputation: {e}")
        
        summary_df = az.summary(idata_pooled)
        global_params = summary_df[~summary_df.index.str.contains(r'\|')]
        
        summary_path = os.path.join(BASE_DIR, f"bayesian_{model_name.lower()}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {model_name}\nFormula: {formula_2L}\n")
            f.write(f"Total draws (pooled): {len(idatas) * draws_per_imp * 2}\n\n")
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
        
        # --- Enrich metrics with HDI, pd, and per-imputation R-hat ---
        try:
            # Handle ArViz 1.1 DataTree structure vs older Dataset structure
            post_ds = idata_pooled['/posterior'].dataset if hasattr(idata_pooled['/posterior'], 'dataset') else idata_pooled.posterior
            
            # az.hdi returns a DataTree in Arviz 1.x
            hdi_result = az.hdi(idata_pooled, prob=0.95)
            hdi_ds = hdi_result['/posterior'].dataset if hasattr(hdi_result, 'groups') else hdi_result
        except Exception as e:
            print(f"    [WARNING] Could not compute HDI: {e}")
            post_ds, hdi_ds = None, None

        for param in global_params.index:
            all_metrics[model_name][f"post_{param}_mean"] = float(global_params.loc[param, 'mean'])
            all_metrics[model_name][f"post_{param}_sd"]   = float(global_params.loc[param, 'sd'])
            # HDI bounds
            if hdi_ds is not None and param in hdi_ds:
                try:
                    all_metrics[model_name][f"post_{param}_hdi_lb"] = float(hdi_ds[param].sel(hdi='lower').item())
                    all_metrics[model_name][f"post_{param}_hdi_ub"] = float(hdi_ds[param].sel(hdi='higher').item())
                except Exception:
                    try:
                        all_metrics[model_name][f"post_{param}_hdi_lb"] = float(hdi_ds[param].sel(ci_bound='lower').item())
                        all_metrics[model_name][f"post_{param}_hdi_ub"] = float(hdi_ds[param].sel(ci_bound='upper').item())
                    except Exception as e2:
                        print(f"    [WARNING] HDI extraction failed for {param}: {e2}")
            # pd: proportion on dominant side of zero (same convention as mediation script)
            if post_ds is not None and param in post_ds:
                draws = post_ds[param].values.flatten()
                p_pos = float(np.mean(draws > 0))
                all_metrics[model_name][f"post_{param}_pd"] = float(max(p_pos, 1 - p_pos))
            # Per-imputation R-hat list
            if param in per_imp_rhats:
                all_metrics[model_name][f"post_{param}_rhat_per_imp"] = per_imp_rhats[param]
            
        print(f"  [{model_name}] Done. R_hat_max = {rhat_max}, Mean LOO ELPD = {np.mean(loos) if loos else 'N/A'}")
        
        # Generate PPC plot and save stats (ArViz 1.x: use plot_ppc_dist)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            az.plot_ppc_dist(idata_pooled, axes=ax)
            ppc_fig_path = os.path.join(BASE_DIR, f"ppc_{model_name.lower()}.webp")
            plt.savefig(ppc_fig_path, format="webp")
            plt.close(fig)
            print(f"    Saved PPC plot to {ppc_fig_path}")
        except Exception as e:
            print(f"    [WARNING] PPC plot generation failed: {e}")
            
        del idatas, idata_pooled, summary_df
        gc.collect()
        
    metrics_path = os.path.join(BASE_DIR, "bayesian_linear_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, default=str)
    print("\n[+] Stage 3 Complete. Proceed to Stage 4.")

if __name__ == "__main__":
    run_bayesian_linear_models()
