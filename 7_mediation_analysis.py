import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

from stats_utils import BASE_DIR, DATASETS, apply_rubins_rules

REPORT_OUTPUT = os.path.join(BASE_DIR, 'mediation_report.txt')
DATASET_PATH = DATASETS['imputed_passive']

F_PATH_A_WP = 'Leisure_WP_z ~ Workload_WP_z + Sleep_WP_z + day_number_z + is_weekend'
F_PATH_A_BP = 'Leisure_BP_z ~ Workload_BP_z + Sleep_BP_z + Trait_Fatigue_z'

F_PATH_BC_WP = (
    'Next_Day_Fatigue ~ Workload_WP_z + Next_Day_Workload_WP_z + Leisure_WP_z + Sleep_WP_z + '
    'Workload_BP_z + Leisure_BP_z + Sleep_BP_z + Trait_Fatigue_z + day_number_z + is_weekend'
)

F_PATH_C_TOTAL = (
    'Next_Day_Fatigue ~ Workload_WP_z + Next_Day_Workload_WP_z + Sleep_WP_z + '
    'Workload_BP_z + Sleep_BP_z + Trait_Fatigue_z + day_number_z + is_weekend'
)

F_PLACEBO = (
    'Next_Day_Leisure ~ S2_Fatigue_Mean_z + Sleep_WP_z + '
    'Workload_BP_z + Sleep_BP_z + Trait_Fatigue_z + day_number_z + is_weekend'
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
        imputations.append((imp, df_imp))
    return imputations

def fit_mixedlm(formula, df):
    try: return smf.mixedlm(formula, df, groups=df['Participant_No']).fit(reml=True, method='lbfgs')
    except Exception: return smf.mixedlm(formula, df, groups=df['Participant_No']).fit(reml=True, method='bfgs')

def fit_ols(formula, df):
    df_bp = df.drop_duplicates(subset=['Participant_No'])
    return smf.ols(formula, data=df_bp).fit()

def run_imputation_loop(imputations, formula, is_bp=False):
    ests, bses, ns, n_parts = [], [], [], []
    for imp, df_imp in imputations:
        try:
            if is_bp:
                res = fit_ols(formula, df_imp)
                ests.append(dict(res.params)); bses.append(dict(res.bse)); ns.append(int(res.nobs)); n_parts.append(int(res.nobs))
            else:
                res = fit_mixedlm(formula, df_imp)
                ests.append(dict(res.fe_params)); bses.append(dict(res.bse_fe)); ns.append(int(res.nobs)); n_parts.append(df_imp['Participant_No'].nunique())
        except Exception as e:
            pass
    level_mapping = {p: 2 for p in ests[0].keys()} if is_bp and ests else None
    pooled = apply_rubins_rules(ests, bses, ns, n_parts, level_mapping=level_mapping)
    return pooled, np.mean(ns), np.mean(n_parts)

def monte_carlo_indirect_effect(a_coef, a_se, b_coef, b_se, n_sims=20000, seed=42):
    rng = np.random.default_rng(seed)
    a_draws = rng.normal(a_coef, a_se, n_sims)
    b_draws = rng.normal(b_coef, b_se, n_sims)
    indirect_draws = a_draws * b_draws
    point_estimate = a_coef * b_coef
    ci_lower, ci_upper = np.percentile(indirect_draws, [2.5, 97.5])
    return {
        'indirect_effect': point_estimate, 'ci_95_lower': ci_lower, 'ci_95_upper': ci_upper,
        'significant': not (ci_lower < 0 < ci_upper),
        'a_coef': a_coef, 'a_se': a_se, 'b_coef': b_coef, 'b_se': b_se
    }

def safe_format(val, dec=4):
    if pd.isna(val) or val is None: return 'N/A'
    return f"{val:.{dec}f}"

def run_mediation_analysis():
    print('=' * 70 + '\n  PIPELINE STAGE 7: LAGGED MEDIATION ANALYSIS\n' + '=' * 70)
    df_full = pd.read_parquet(DATASET_PATH)
    imputations = prepare_lagged_data(df_full)
    print(f"  Prepared m={len(imputations)} imputations. Mean N = {int(len(imputations[0][1]))} consecutive days.")
    
    pool_a_wp, n_obs, _ = run_imputation_loop(imputations, F_PATH_A_WP)
    pool_a_bp, _, _ = run_imputation_loop(imputations, F_PATH_A_BP, is_bp=True)
    pool_bc, _, _ = run_imputation_loop(imputations, F_PATH_BC_WP)
    pool_c, _, _ = run_imputation_loop(imputations, F_PATH_C_TOTAL)
    pool_placebo, _, _ = run_imputation_loop(imputations, F_PLACEBO)
    
    med_wp = monte_carlo_indirect_effect(pool_a_wp['Workload_WP_z']['Estimate'], pool_a_wp['Workload_WP_z']['SE'], pool_bc['Leisure_WP_z']['Estimate'], pool_bc['Leisure_WP_z']['SE'])
    med_bp = monte_carlo_indirect_effect(pool_a_bp['Workload_BP_z']['Estimate'], pool_a_bp['Workload_BP_z']['SE'], pool_bc['Leisure_BP_z']['Estimate'], pool_bc['Leisure_BP_z']['SE'])
    
    tot_wp = pool_c['Workload_WP_z']['Estimate']
    prop_wp = med_wp['indirect_effect'] / tot_wp if tot_wp != 0 else np.nan
    tot_bp = pool_c['Workload_BP_z']['Estimate']
    prop_bp = med_bp['indirect_effect'] / tot_bp if tot_bp != 0 else np.nan
    
    rep = []
    rep.append('='*70 + '\n  LAGGED MULTILEVEL MEDIATION ANALYSIS REPORT\n' + '='*70)
    rep.append(f"Model N: {int(n_obs)} consecutive-day observations (pooled across m={len(imputations)} imputations)")
    
    rep.append('\n--- WITHIN-PERSON MEDIATION (Daily fluctuations) ---')
    rep.append(f"  Path A (Workload_WP -> Leisure_WP):    Coef = {safe_format(med_wp['a_coef'])}, SE = {safe_format(med_wp['a_se'])}, p = {safe_format(pool_a_wp['Workload_WP_z']['p_val'])}")
    rep.append(f"  Path B (Leisure_WP -> Fatigue_t+1):    Coef = {safe_format(med_wp['b_coef'])}, SE = {safe_format(med_wp['b_se'])}, p = {safe_format(pool_bc['Leisure_WP_z']['p_val'])}")
    rep.append(f"  Total Effect C (Workload_WP alone):    Coef = {safe_format(tot_wp)}, p = {safe_format(pool_c['Workload_WP_z']['p_val'])}")
    rep.append(f"  Direct Effect C' (net of Leisure):     Coef = {safe_format(pool_bc['Workload_WP_z']['Estimate'])}, p = {safe_format(pool_bc['Workload_WP_z']['p_val'])}")
    rep.append(f"\n  INDIRECT EFFECT (a*b):                 {safe_format(med_wp['indirect_effect'])}")
    rep.append(f"  Monte Carlo 95% CI:                    [{safe_format(med_wp['ci_95_lower'])}, {safe_format(med_wp['ci_95_upper'])}]")
    if med_wp['significant'] and med_wp['indirect_effect'] > 0: rep.append('  -> SIGNIFICANT AND POSITIVE: Supports hypothesis.')
    elif med_wp['significant']: rep.append('  -> SIGNIFICANT BUT NEGATIVE: Contra-hypothesis direction.')
    else: rep.append('  -> NULL FINDING: CI includes zero. No evidence for WP mediation.')
    rep.append(f"  Proportion Mediated:                   {safe_format(prop_wp*100,1)}% (Descriptive only)")
    
    rep.append('\n--- BETWEEN-PERSON MEDIATION (Trait-level/Chronic effects) ---')
    rep.append(f"  Path A (Workload_BP -> Leisure_BP):    Coef = {safe_format(med_bp['a_coef'])}, SE = {safe_format(med_bp['a_se'])}, p = {safe_format(pool_a_bp['Workload_BP_z']['p_val'])}")
    rep.append(f"  Path B (Leisure_BP -> Fatigue_t+1):    Coef = {safe_format(med_bp['b_coef'])}, SE = {safe_format(med_bp['b_se'])}, p = {safe_format(pool_bc['Leisure_BP_z']['p_val'])}")
    rep.append(f"  Total Effect C (Workload_BP alone):    Coef = {safe_format(tot_bp)}, p = {safe_format(pool_c['Workload_BP_z']['p_val'])}")
    rep.append(f"  Direct Effect C' (net of Leisure):     Coef = {safe_format(pool_bc['Workload_BP_z']['Estimate'])}, p = {safe_format(pool_bc['Workload_BP_z']['p_val'])}")
    rep.append(f"\n  INDIRECT EFFECT (a*b):                 {safe_format(med_bp['indirect_effect'])}")
    rep.append(f"  Monte Carlo 95% CI:                    [{safe_format(med_bp['ci_95_lower'])}, {safe_format(med_bp['ci_95_upper'])}]")
    if med_bp['significant'] and med_bp['indirect_effect'] > 0: rep.append('  -> SIGNIFICANT AND POSITIVE: Supports hypothesis.')
    elif med_bp['significant']: rep.append('  -> SIGNIFICANT BUT NEGATIVE: Contra-hypothesis direction.')
    else: rep.append('  -> NULL FINDING: CI includes zero. No evidence for BP mediation.')
    rep.append(f"  Proportion Mediated:                   {safe_format(prop_bp*100,1)}% (Descriptive only)")
    
    rep.append('\n--- PLACEBO TEST (Reverse Direction) ---')
    rep.append(f"  Hypothesis: Fatigue(t) -> Leisure_WP(t+1)")
    rep.append(f"  Fatigue Effect:                        Coef = {safe_format(pool_placebo['S2_Fatigue_Mean_z']['Estimate'])}, p = {safe_format(pool_placebo['S2_Fatigue_Mean_z']['p_val'])}")
    
    rt = '\n'.join(rep)
    print('\n' + rt)
    with open(REPORT_OUTPUT, 'w', encoding='utf-8') as f: f.write(rt)

if __name__ == '__main__':
    run_mediation_analysis()
