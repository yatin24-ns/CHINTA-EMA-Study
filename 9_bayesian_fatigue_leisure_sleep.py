import os
import warnings
import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az
import xarray as xr

warnings.filterwarnings('ignore')

from stats_utils import BASE_DIR, DATASETS

REPORT_OUTPUT = os.path.join(BASE_DIR, 'bayesian_leisure_sleep_report.txt')
DATASET_PATH = DATASETS['imputed_passive']

# Path A: Fatigue(t) -> Leisure(t)
F_PATH_A = (
    'Leisure_WP_z ~ S2_Fatigue_Mean_z + Workload_WP_z + Sleep_WP_z + '
    'Workload_BP_z + Sleep_BP_z + Trait_Fatigue_z + day_number_z + is_weekend + (1 | Participant_No)'
)

# Path B & C': Leisure(t) & Fatigue(t) -> Sleep(t+1)
# Note: Sleep_WP_z is the AR(1) control for sleep on day t.
F_PATH_BC = (
    'Next_Day_Sleep ~ Leisure_WP_z + S2_Fatigue_Mean_z + Sleep_WP_z + next_day_is_weekend + '
    'Workload_WP_z + Workload_BP_z + Sleep_BP_z + Trait_Fatigue_z + day_number_z + is_weekend + (1 | Participant_No)'
)

# Path C: Fatigue(t) -> Sleep(t+1)
F_PATH_C = (
    'Next_Day_Sleep ~ S2_Fatigue_Mean_z + Sleep_WP_z + next_day_is_weekend + '
    'Workload_WP_z + Workload_BP_z + Sleep_BP_z + Trait_Fatigue_z + day_number_z + is_weekend + (1 | Participant_No)'
)

# Placebo: Leisure(t) -> Sleep(t) (Predicting the past)
F_PLACEBO = (
    'Sleep_WP_z ~ Leisure_WP_z + Workload_WP_z + Workload_BP_z + '
    'Trait_Fatigue_z + day_number_z + is_weekend + (1 | Participant_No)'
)

def prepare_data(df_full):
    imputations = []
    for imp in sorted(df_full['.imp'].unique()):
        df_imp = df_full[df_full['.imp'] == imp].sort_values(['Participant_No', 'day_number']).copy()
        
        # Next Day constructions
        df_imp['Next_Day_Sleep'] = df_imp.groupby('Participant_No')['Sleep_WP_z'].shift(-1)
        df_imp['next_day_is_weekend'] = df_imp.groupby('Participant_No')['is_weekend'].shift(-1)
        
        # Rigorous day gap validation
        df_imp['day_gap'] = df_imp.groupby('Participant_No')['day_number'].diff(-1).abs()
        df_imp = df_imp[df_imp['day_gap'] == 1].copy()
        
        # Drop rows with missing values in our key variables
        req_cols = [
            'Next_Day_Sleep', 'Leisure_WP_z', 'S2_Fatigue_Mean_z', 'Sleep_WP_z', 
            'Workload_WP_z', 'next_day_is_weekend'
        ]
        df_imp = df_imp.dropna(subset=req_cols)
        
        # Convert Participant_No to string for bambi categorical grouping
        df_imp['Participant_No'] = df_imp['Participant_No'].astype(str)
        imputations.append((imp, df_imp))
    return imputations

def concat_idatas(idatas):
    if len(idatas) == 1: return idatas[0]
    all_groups = set()
    for idata in idatas:
        for g in idata.groups:
            if g != '/': all_groups.add(g.lstrip('/'))
    tree_dict = {}
    for group in all_groups:
        group_datasets = []
        for idata in idatas:
            try:
                node = idata[f"/{group}"]
                ds = node.ds
                if ds is not None and len(ds) > 0: group_datasets.append(ds)
            except (KeyError, ValueError): pass
        if not group_datasets: continue
        if group == "observed_data":
            tree_dict[f"/{group}"] = group_datasets[0]
        else:
            try: tree_dict[f"/{group}"] = xr.concat(group_datasets, dim="chain")
            except Exception: tree_dict[f"/{group}"] = group_datasets[0]
    return xr.DataTree.from_dict(tree_dict)

def fit_bambi_models(formula, imputations, draws=500, tune=500):
    idatas = []
    ns = []
    for imp, df_imp in imputations:
        ns.append(len(df_imp))
        try:
            model = bmb.Model(formula, df_imp)
            idata = model.fit(draws=draws, tune=tune, chains=2, cores=1, target_accept=0.90, random_seed=42+imp)
            idatas.append(idata)
            del model
        except Exception as e:
            print(f"  [ERROR] Fitting failed for imp {imp}: {e}")
    
    if not idatas: return None, 0
    pooled_idata = concat_idatas(idatas)
    return pooled_idata, np.mean(ns)

def get_post_stats(posterior, var_name):
    data = posterior[var_name].values.flatten()
    mean = np.mean(data)
    hdi_3, hdi_97 = np.percentile(data, [2.5, 97.5])
    prob_gt_0 = np.mean(data > 0)
    prob_lt_0 = np.mean(data < 0)
    p_dir = max(prob_gt_0, prob_lt_0)
    return mean, hdi_3, hdi_97, p_dir

def safe_format(val, dec=4):
    if pd.isna(val) or val is None: return 'N/A'
    return f"{val:.{dec}f}"

def run_analysis():
    print('=' * 70 + '\n  PIPELINE STAGE 9: BAYESIAN FATIGUE -> LEISURE -> SLEEP MEDIATION\n' + '=' * 70)
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Run Stage 1 first.")
        return
        
    df_full = pd.read_parquet(DATASET_PATH)
    imputations = prepare_data(df_full)
    print(f"  Prepared m={len(imputations)} imputations. Mean N = {int(len(imputations[0][1]))} consecutive days.")
    print("  Fitting Bambi models (this will take some time)...")
    
    draws, tune = 500, 500
    
    print("\n  [1/4] Fitting Path A (Fatigue -> Leisure)...")
    idata_a, n_obs = fit_bambi_models(F_PATH_A, imputations, draws=draws, tune=tune)
    
    print("  [2/4] Fitting Path B/C' (Leisure & Fatigue -> Next Sleep)...")
    idata_bc, _ = fit_bambi_models(F_PATH_BC, imputations, draws=draws, tune=tune)
    
    print("  [3/4] Fitting Path C (Fatigue -> Next Sleep Total Effect)...")
    idata_c, _ = fit_bambi_models(F_PATH_C, imputations, draws=draws, tune=tune)
    
    print("  [4/4] Fitting Placebo (Leisure -> Prev Sleep)...")
    idata_placebo, _ = fit_bambi_models(F_PLACEBO, imputations, draws=draws, tune=tune)
    
    if not all([idata_a, idata_bc, idata_c, idata_placebo]):
        print("One or more models failed to fit. Aborting reporting.")
        return
        
    post_a = idata_a.posterior['S2_Fatigue_Mean_z']
    post_b = idata_bc.posterior['Leisure_WP_z']
    post_cprime = idata_bc.posterior['S2_Fatigue_Mean_z']
    post_ctot = idata_c.posterior['S2_Fatigue_Mean_z']
    
    # Exact joint posterior of indirect effect
    indirect = post_a * post_b
    
    # Proportion mediated
    prop = (indirect / post_ctot).values.flatten()
    prop_med = np.median(prop[(prop > -2) & (prop < 2)]) if len(prop) > 0 else np.nan
    
    def report_line(name, posterior, var_name=None):
        if var_name:
            m, l, u, pdir = get_post_stats(posterior, var_name)
        else:
            data = posterior.values.flatten()
            m = np.mean(data)
            l, u = np.percentile(data, [2.5, 97.5])
            pdir = max(np.mean(data > 0), np.mean(data < 0))
        return f"{name:45s} Coef = {safe_format(m)}, 95% HDI = [{safe_format(l)}, {safe_format(u)}], pd = {safe_format(pdir*100, 1)}%"

    rep = []
    rep.append('='*75 + '\n  BAYESIAN LAGGED MEDIATION: FATIGUE(t) -> LEISURE(t) -> SLEEP(t+1)\n' + '='*75)
    rep.append(f"Model N: ~{int(n_obs)} consecutive-day observations (pooled across m={len(imputations)} imputations)")
    rep.append(f"AR(1) Autocorrelation controlled via Sleep_WP_z(t) in all models.")
    
    rep.append('\n--- MEDIATION PATHS ---')
    rep.append("  " + report_line("Path A (Fatigue_t -> Leisure_t):", idata_a.posterior, 'S2_Fatigue_Mean_z'))
    rep.append("  " + report_line("Path B (Leisure_t -> Sleep_t+1):", idata_bc.posterior, 'Leisure_WP_z'))
    rep.append("  " + report_line("Total Effect C (Fatigue_t alone):", idata_c.posterior, 'S2_Fatigue_Mean_z'))
    rep.append("  " + report_line("Direct Effect C' (net of Leisure):", idata_bc.posterior, 'S2_Fatigue_Mean_z'))
    
    rep.append(f"\n  " + report_line("INDIRECT EFFECT (a*b):", indirect))
    sig = np.percentile(indirect.values.flatten(), 2.5) > 0 or np.percentile(indirect.values.flatten(), 97.5) < 0
    dir_str = "POSITIVE" if np.mean(indirect.values.flatten()) > 0 else "NEGATIVE"
    if sig: rep.append(f'  -> SIGNIFICANT AND {dir_str}: 95% HDI excludes zero.')
    else: rep.append('  -> NULL FINDING: 95% HDI includes zero. No evidence for mediation.')
    rep.append(f"  Proportion Mediated (Median):               {safe_format(prop_med*100,1)}% (Descriptive only)")
    
    rep.append('\n--- FALSIFICATION / PLACEBO TEST (Reverse Time) ---')
    rep.append("  Hypothesis: Leisure(t) -> Sleep(t) [Testing if trait-level copers confound day-to-day causality]")
    rep.append("  " + report_line("Placebo Effect:", idata_placebo.posterior, 'Leisure_WP_z'))
    
    rt = '\n'.join(rep)
    print('\n' + rt)
    with open(REPORT_OUTPUT, 'w', encoding='utf-8') as f: f.write(rt)

if __name__ == '__main__':
    run_analysis()
