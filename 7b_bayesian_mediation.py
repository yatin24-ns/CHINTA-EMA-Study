import os
import warnings
import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az
import xarray as xr
import gc

warnings.filterwarnings('ignore')

from stats_utils import BASE_DIR, DATASETS

REPORT_OUTPUT = os.path.join(BASE_DIR, 'bayesian_mediation_report.txt')
DATASET_PATH = DATASETS['imputed_passive']

F_PATH_A_WP = 'Leisure_WP_z ~ Workload_WP_z + Sleep_WP_z + day_number_z + is_weekend + (1 | Participant_No)'
F_PATH_A_BP = 'Leisure_BP_z ~ Workload_BP_z + Sleep_BP_z + Trait_Fatigue_z'

F_PATH_BC_WP = (
    'Next_Day_Fatigue ~ Workload_WP_z + Next_Day_Workload_WP_z + Leisure_WP_z + Sleep_WP_z + '
    'Workload_BP_z + Leisure_BP_z + Sleep_BP_z + Trait_Fatigue_z + day_number_z + is_weekend + (1 | Participant_No)'
)

F_PATH_C_TOTAL = (
    'Next_Day_Fatigue ~ Workload_WP_z + Next_Day_Workload_WP_z + Sleep_WP_z + '
    'Workload_BP_z + Sleep_BP_z + Trait_Fatigue_z + day_number_z + is_weekend + (1 | Participant_No)'
)

F_PLACEBO = (
    'Next_Day_Leisure ~ S2_Fatigue_Mean_z + Sleep_WP_z + '
    'Workload_BP_z + Sleep_BP_z + Trait_Fatigue_z + day_number_z + is_weekend + (1 | Participant_No)'
)

def prepare_lagged_data(df_full):
    imputations = []
    for imp in sorted(df_full['.imp'].unique()):
        df_imp = df_full[df_full['.imp'] == imp].sort_values(['Participant_No', 'day_number']).copy()
        df_imp['Next_Day_Fatigue'] = df_imp.groupby('Participant_No')['S2_Fatigue_Mean_z'].shift(-1)
        df_imp['Next_Day_Leisure'] = df_imp.groupby('Participant_No')['Leisure_WP_z'].shift(-1)
        df_imp['Next_Day_Workload_WP_z'] = df_imp.groupby('Participant_No')['Workload_WP_z'].shift(-1)
        df_imp['day_gap'] = df_imp.groupby('Participant_No')['day_number'].diff(-1).abs()
        df_imp = df_imp[df_imp['day_gap'] == 1].copy()
        df_imp = df_imp.dropna(subset=['Next_Day_Fatigue', 'Next_Day_Leisure', 'Next_Day_Workload_WP_z', 'S2_Fatigue_Mean_z', 'Workload_WP_z', 'Leisure_WP_z'])
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

def fit_bambi_models(formula, imputations, is_bp=False, draws=500, tune=500):
    idatas = []
    ns = []
    for imp, df_imp in imputations:
        if is_bp:
            df_fit = df_imp.drop_duplicates(subset=['Participant_No'])
        else:
            df_fit = df_imp
        
        ns.append(len(df_fit))
        try:
            model = bmb.Model(formula, df_fit)
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

def run_bayesian_mediation():
    print('=' * 70 + '\n  PIPELINE STAGE 7b: BAYESIAN LAGGED MEDIATION ANALYSIS\n' + '=' * 70)
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Run Stage 1 first.")
        return
        
    df_full = pd.read_parquet(DATASET_PATH)
    imputations = prepare_lagged_data(df_full)
    print(f"  Prepared m={len(imputations)} imputations. Mean N = {int(len(imputations[0][1]))} consecutive days.")
    print("  Fitting Bambi models (this will take some time)...")
    
    draws, tune = 500, 500
    
    print("\n  [1/5] Fitting Path A (Within-Person)...")
    idata_a_wp, n_wp = fit_bambi_models(F_PATH_A_WP, imputations, draws=draws, tune=tune)
    
    print("  [2/5] Fitting Path A (Between-Person)...")
    idata_a_bp, n_bp = fit_bambi_models(F_PATH_A_BP, imputations, is_bp=True, draws=draws, tune=tune)
    
    print("  [3/5] Fitting Path B/C' (Direct Effects)...")
    idata_bc, _ = fit_bambi_models(F_PATH_BC_WP, imputations, draws=draws, tune=tune)
    
    print("  [4/5] Fitting Path C (Total Effect)...")
    idata_c, _ = fit_bambi_models(F_PATH_C_TOTAL, imputations, draws=draws, tune=tune)
    
    print("  [5/5] Fitting Placebo Test...")
    idata_placebo, _ = fit_bambi_models(F_PLACEBO, imputations, draws=draws, tune=tune)
    
    if not all([idata_a_wp, idata_a_bp, idata_bc, idata_c, idata_placebo]):
        print("One or more models failed to fit. Aborting reporting.")
        return
        
    post_a_wp = idata_a_wp.posterior['Workload_WP_z']
    post_a_bp = idata_a_bp.posterior['Workload_BP_z']
    
    post_b_wp = idata_bc.posterior['Leisure_WP_z']
    post_b_bp = idata_bc.posterior['Leisure_BP_z']
    
    post_cprime_wp = idata_bc.posterior['Workload_WP_z']
    post_cprime_bp = idata_bc.posterior['Workload_BP_z']
    
    post_ctot_wp = idata_c.posterior['Workload_WP_z']
    post_ctot_bp = idata_c.posterior['Workload_BP_z']
    
    indirect_wp = post_a_wp * post_b_wp
    indirect_bp = post_a_bp * post_b_bp
    
    prop_wp = (indirect_wp / post_ctot_wp).values.flatten()
    prop_bp = (indirect_bp / post_ctot_bp).values.flatten()
    prop_wp_med = np.median(prop_wp[(prop_wp > -2) & (prop_wp < 2)]) if len(prop_wp) > 0 else np.nan
    prop_bp_med = np.median(prop_bp[(prop_bp > -2) & (prop_bp < 2)]) if len(prop_bp) > 0 else np.nan
    
    def report_line(name, posterior, var_name=None):
        if var_name:
            m, l, u, pdir = get_post_stats(posterior, var_name)
        else:
            data = posterior.values.flatten()
            m = np.mean(data)
            l, u = np.percentile(data, [2.5, 97.5])
            pdir = max(np.mean(data > 0), np.mean(data < 0))
        return f"{name:40s} Coef = {safe_format(m)}, 95% HDI = [{safe_format(l)}, {safe_format(u)}], pd = {safe_format(pdir*100, 1)}%"

    rep = []
    rep.append('='*70 + '\n  BAYESIAN LAGGED MULTILEVEL MEDIATION ANALYSIS REPORT\n' + '='*70)
    rep.append(f"Model N (WP): ~{int(n_wp)} consecutive-day observations (pooled across m={len(imputations)} imputations)")
    rep.append(f"Model N (BP): ~{int(n_bp)} participants (pooled across m={len(imputations)} imputations)")
    
    rep.append('\n--- WITHIN-PERSON MEDIATION (Daily fluctuations) ---')
    rep.append("  " + report_line("Path A (Workload_WP -> Leisure_WP):", idata_a_wp.posterior, 'Workload_WP_z'))
    rep.append("  " + report_line("Path B (Leisure_WP -> Fatigue_t+1):", idata_bc.posterior, 'Leisure_WP_z'))
    rep.append("  " + report_line("Total Effect C (Workload_WP alone):", idata_c.posterior, 'Workload_WP_z'))
    rep.append("  " + report_line("Direct Effect C' (net of Leisure):", idata_bc.posterior, 'Workload_WP_z'))
    
    rep.append(f"\n  " + report_line("INDIRECT EFFECT (a*b):", indirect_wp))
    sig_wp = np.percentile(indirect_wp.values.flatten(), 2.5) > 0 or np.percentile(indirect_wp.values.flatten(), 97.5) < 0
    dir_wp = "POSITIVE" if np.mean(indirect_wp.values.flatten()) > 0 else "NEGATIVE"
    if sig_wp: rep.append(f'  -> SIGNIFICANT AND {dir_wp}: 95% HDI excludes zero.')
    else: rep.append('  -> NULL FINDING: 95% HDI includes zero. No evidence for WP mediation.')
    rep.append(f"  Proportion Mediated (Median):          {safe_format(prop_wp_med*100,1)}% (Descriptive only)")
    
    rep.append('\n--- BETWEEN-PERSON MEDIATION (Trait-level/Chronic effects) ---')
    rep.append("  " + report_line("Path A (Workload_BP -> Leisure_BP):", idata_a_bp.posterior, 'Workload_BP_z'))
    rep.append("  " + report_line("Path B (Leisure_BP -> Fatigue_t+1):", idata_bc.posterior, 'Leisure_BP_z'))
    rep.append("  " + report_line("Total Effect C (Workload_BP alone):", idata_c.posterior, 'Workload_BP_z'))
    rep.append("  " + report_line("Direct Effect C' (net of Leisure):", idata_bc.posterior, 'Workload_BP_z'))
    
    rep.append(f"\n  " + report_line("INDIRECT EFFECT (a*b):", indirect_bp))
    sig_bp = np.percentile(indirect_bp.values.flatten(), 2.5) > 0 or np.percentile(indirect_bp.values.flatten(), 97.5) < 0
    dir_bp = "POSITIVE" if np.mean(indirect_bp.values.flatten()) > 0 else "NEGATIVE"
    if sig_bp: rep.append(f'  -> SIGNIFICANT AND {dir_bp}: 95% HDI excludes zero.')
    else: rep.append('  -> NULL FINDING: 95% HDI includes zero. No evidence for BP mediation.')
    rep.append(f"  Proportion Mediated (Median):          {safe_format(prop_bp_med*100,1)}% (Descriptive only)")
    
    rep.append('\n--- PLACEBO TEST (Reverse Direction) ---')
    rep.append("  Hypothesis: Fatigue(t) -> Leisure_WP(t+1)")
    rep.append("  " + report_line("Fatigue Effect:", idata_placebo.posterior, 'S2_Fatigue_Mean_z'))
    
    rt = '\n'.join(rep)
    print('\n' + rt)
    with open(REPORT_OUTPUT, 'w', encoding='utf-8') as f: f.write(rt)

if __name__ == '__main__':
    run_bayesian_mediation()
