import pandas as pd
import numpy as np
import scipy.stats
import os

BASE_DIR = r"C:\Users\praka\OneDrive\Desktop\CHINTA practice"
FORMATTED_DIR = os.path.join(BASE_DIR, "Formatted_Data")
IMPUTE_DIR    = os.path.join(FORMATTED_DIR, "Imputation_Datasets")

DATASETS = {
    "original":       os.path.join(FORMATTED_DIR, "original_complete_cases.xlsx"),
    "imputed_jav":    os.path.join(IMPUTE_DIR, "imputed_processed_ema_jav.parquet"),
    "imputed_passive": os.path.join(IMPUTE_DIR, "imputed_processed_ema_passive.parquet"),
    "imputed":        os.path.join(IMPUTE_DIR, "imputed_processed_ema.parquet"), # Backward compatibility
}

def apply_rubins_rules(estimates_list, bse_list, n_obs_list, n_participants_list, level_mapping=None):
    """
    Applies Rubin's Rules to pool parameters across m datasets.
    """
    m = len(estimates_list)
    if m == 0:
        return {}
        
    param_names = estimates_list[0].keys()
    
    pooled = {}
    for p in param_names:
        ests = [d[p] for d in estimates_list if p in d]
        ses = [d[p] for d in bse_list if p in d]
        
        if len(ests) < m:
            continue
            
        theta_bar = np.mean(ests)
        W = np.mean([se**2 for se in ses]) # Within variance
        B = np.var(ests, ddof=1) if m > 1 else 0 # Between variance
        T = W + (1 + 1/m) * B # Total variance
        se_pooled = np.sqrt(T)
        
        # Barnard-Rubin adjusted degrees of freedom
        if B > 0 and W > 0:
            r = (1 + 1/m) * (B / W)
            v_old = (m - 1) * (1 + 1/r)**2
            
            # Dynamic df for Level-2 vs Level-1
            n_total = np.mean(n_obs_list)
            n_participants = np.mean(n_participants_list)
            k = len(param_names)
            
            if level_mapping:
                # Use explicit mapping if provided
                is_level2 = (level_mapping.get(p, 1) == 2)
            else:
                # Fallback to string matching
                is_level2 = ('_BP' in p or 'Trait' in p or p == 'Intercept')
                
            if is_level2:
                v_com_raw = n_participants - k
            else:
                v_com_raw = n_total - k
                
            if v_com_raw <= 0:
                print(f"  [WARNING] v_com <= 0 for {p} (v_com_raw={v_com_raw}). Clamping to 1.")
            v_com = max(1, v_com_raw)
                
            gamma = (1 + 1/m) * B / T
            v_obs = ((v_com + 1) / (v_com + 3)) * v_com * (1 - gamma)
            df = (v_old * v_obs) / (v_old + v_obs) if (v_old + v_obs) > 0 else v_old
        else:
            df = float('inf')
            
        t_stat = theta_bar / se_pooled if se_pooled > 0 else 0
        p_val = scipy.stats.t.sf(np.abs(t_stat), df) * 2
        
        pooled[p] = {
            "Estimate": round(theta_bar, 4),
            "SE": round(se_pooled, 4),
            "t_or_z": round(t_stat, 4),
            "p_val": round(p_val, 4)
        }
        
    return pooled

def extract_bambi_metrics(idata_pooled, model_name, dataset_name, m_imputations, nc_path, rhat_threshold=1.05):
    """
    Extracts summary statistics and convergence metrics from a pooled InferenceData object.
    Saves a text summary to disk and returns a dictionary of metrics including parameter estimates.
    """
    import arviz as az
    import pandas as pd
    
    summary_df = az.summary(idata_pooled)
    global_params = summary_df[~summary_df.index.str.contains(r'\|')]
    
    summary_txt_path = os.path.join(BASE_DIR, f"bayesian_{model_name.lower()}_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_df.to_string())
        
    rhat_max = pd.to_numeric(summary_df['r_hat'], errors='coerce').max() if 'r_hat' in summary_df.columns else None
    
    metrics = {
        "Dataset": dataset_name, 
        "Converged": bool(rhat_max < rhat_threshold) if pd.notna(rhat_max) else False,
        "R_hat_max": float(rhat_max) if pd.notna(rhat_max) else None, 
        "Trace_file": nc_path, 
        "m_imputations": m_imputations,
        "Parameters": {}
    }
    
    for param in global_params.index:
        metrics["Parameters"][param] = {
            "Estimate": float(global_params.loc[param, 'mean']),
            "SE": float(global_params.loc[param, 'sd'])
        }
        
    return metrics
