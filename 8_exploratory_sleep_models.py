"""
PIPELINE STAGE 8: EXPLORATORY SLEEP MODELS
Follow-up to Stage 7 (Mediation Analysis). Tests same-day moderation and dose-response for Sleep.
"""
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

from stats_utils import BASE_DIR, DATASETS, apply_rubins_rules

REPORT_OUTPUT = os.path.join(BASE_DIR, 'sleep_exploratory_report.txt')
DATASET_PATH_ORIG = DATASETS['original']
DATASET_PATH_IMP_PASSIVE = DATASETS['imputed_passive']
DATASET_PATH_IMP_JAV = DATASETS['imputed_jav']

F_MODERATION = (
    'S2_Fatigue_Mean_z ~ Sleep_WP_z + Workload_WP_z + Sleep_WP_z:Workload_WP_z + '
    'Leisure_WP_z + Workload_BP_z + Sleep_BP_z + Leisure_BP_z + '
    'Trait_Fatigue_z + day_number_z + is_weekend'
)

F_DOSE_RESPONSE = (
    'S2_Fatigue_Mean_z ~ Sleep_WP_z + I(Sleep_WP_z**2) + Workload_WP_z + '
    'Leisure_WP_z + Workload_BP_z + Sleep_BP_z + Leisure_BP_z + '
    'Trait_Fatigue_z + day_number_z + is_weekend'
)

def safe_format(val, dec=4):
    if pd.isna(val) or val is None: return 'N/A'
    return f"{val:.{dec}f}"

def fit_and_pool_model(datasets_to_run, formula):
    ests_ml, bse_ml, ests_reml, bse_reml = [], [], [], []
    n_obs_list, n_participants_list = [], []

    for imp, df_imp in datasets_to_run:
        if df_imp is None or len(df_imp) == 0:
            continue
        try:
            model_ml = smf.mixedlm(formula, df_imp, groups=df_imp['Participant_No'])
            try: res_ml = model_ml.fit(reml=False, method='lbfgs')
            except Exception: res_ml = model_ml.fit(reml=False, method='bfgs')

            model_reml = smf.mixedlm(formula, df_imp, groups=df_imp['Participant_No'])
            try: res_reml = model_reml.fit(reml=True, method='lbfgs')
            except Exception: res_reml = model_reml.fit(reml=True, method='bfgs')

            ests_ml.append(res_ml.fe_params.to_dict())
            bse_ml.append(res_ml.bse_fe.to_dict())
            ests_reml.append(res_reml.fe_params.to_dict())
            bse_reml.append(res_reml.bse_fe.to_dict())
            n_obs_list.append(int(res_ml.nobs))
            n_participants_list.append(df_imp['Participant_No'].nunique())
        except Exception as e:
            print(f"  [ERROR] Imputation {imp} failed: {e}")
            continue

    if not ests_ml: return None, None, None
    pooled_ml = apply_rubins_rules(ests_ml, bse_ml, n_obs_list, n_participants_list)
    pooled_reml = apply_rubins_rules(ests_reml, bse_reml, n_obs_list, n_participants_list)
    n_info = {"n_obs_mean": np.mean(n_obs_list), "n_participants_mean": np.mean(n_participants_list)}
    return pooled_ml, pooled_reml, n_info

def run_exploratory_sleep_models():
    print('=' * 70 + '\n  PIPELINE STAGE 8: EXPLORATORY SLEEP MODELS\n' + '=' * 70)
    
    df_orig = pd.read_excel(DATASET_PATH_ORIG)
    df_imp_passive = pd.read_parquet(DATASET_PATH_IMP_PASSIVE)
    df_imp_jav = pd.read_parquet(DATASET_PATH_IMP_JAV)
    
    # Missing Columns check
    cols_to_drop = ['S2_Fatigue_Mean_z', 'Sleep_WP_z', 'Workload_WP_z', 'Leisure_WP_z', 'Workload_BP_z', 'Sleep_BP_z', 'Leisure_BP_z', 'Trait_Fatigue_z', 'day_number_z', 'is_weekend']
    missing_orig = [c for c in cols_to_drop if c not in df_orig.columns]
    if missing_orig:
        print(f"  [ERROR] Original dataset missing columns: {missing_orig}")
    
    # Multicollinearity Check per imputation (Passive)
    corrs_pass = []
    for imp_val in sorted(df_imp_passive['.imp'].unique()):
        sub = df_imp_passive[df_imp_passive['.imp'] == imp_val]
        c = sub[['Sleep_WP_z']].assign(sq=sub['Sleep_WP_z']**2).corr().iloc[0,1]
        corrs_pass.append(c)
    sleep_corr_pass = np.mean(corrs_pass)
    print(f"  [VALIDATION] Sleep_WP_z / Sleep_WP_z^2 mean correlation across PASSIVE imputations: {sleep_corr_pass:.3f}")
    if abs(sleep_corr_pass) > 0.9:
        print("  [WARNING] High correlation (>0.9) detected between linear and quadratic terms (Passive).")
        
    # Multicollinearity Check per imputation (JAV)
    corrs_jav = []
    for imp_val in sorted(df_imp_jav['.imp'].unique()):
        sub = df_imp_jav[df_imp_jav['.imp'] == imp_val]
        c = sub[['Sleep_WP_z']].assign(sq=sub['Sleep_WP_z']**2).corr().iloc[0,1]
        corrs_jav.append(c)
    sleep_corr_jav = np.mean(corrs_jav)
    print(f"  [VALIDATION] Sleep_WP_z / Sleep_WP_z^2 mean correlation across JAV imputations: {sleep_corr_jav:.3f}")
    if abs(sleep_corr_jav) > 0.9:
        print("  [WARNING] High correlation (>0.9) detected between linear and quadratic terms (JAV).")
    
    orig_datasets = [(0, df_orig.dropna(subset=cols_to_drop))]
    imp_datasets_passive = [(imp, df_imp_passive[df_imp_passive['.imp'] == imp].dropna(subset=cols_to_drop)) for imp in sorted(df_imp_passive['.imp'].unique())]
    imp_datasets_jav = [(imp, df_imp_jav[df_imp_jav['.imp'] == imp].dropna(subset=cols_to_drop)) for imp in sorted(df_imp_jav['.imp'].unique())]
    
    print('\n Fitting Moderation Model (Original)' )
    _, orig_reml_mod, n_orig_mod = fit_and_pool_model(orig_datasets, F_MODERATION)
    print('Fitting Moderation Model (Imputed Passive)')
    _, imp_reml_mod_passive, n_imp_mod_passive = fit_and_pool_model(imp_datasets_passive, F_MODERATION)
    print('Fitting Moderation Model (Imputed JAV)')
    _, imp_reml_mod_jav, n_imp_mod_jav = fit_and_pool_model(imp_datasets_jav, F_MODERATION)
    
    print('\nFitting Dose-Response Model (Original)')
    _, orig_reml_dose, n_orig_dose = fit_and_pool_model(orig_datasets, F_DOSE_RESPONSE)
    print('Fitting Dose-Response Model (Imputed Passive) ')
    _, imp_reml_dose_passive, n_imp_dose_passive = fit_and_pool_model(imp_datasets_passive, F_DOSE_RESPONSE)
    print('Fitting Dose-Response Model (Imputed JAV) ')
    _, imp_reml_dose_jav, n_imp_dose_jav = fit_and_pool_model(imp_datasets_jav, F_DOSE_RESPONSE)
    
    # Reporting
    rep = []
    rep.append('='*70 + '\n  EXPLORATORY SLEEP MODELS REPORT\n' + '='*70)
    rep.append('Note: This concludes the exploratory phase. Any further hypotheses arising from these two tests will be flagged as directions for future/pre-registered study, not tested further within this dataset.\n')
    
    rep.append('TEST 1: SAME-DAY MODERATION (Workload x Sleep)')
    rep.append("Question: Does today's workload blunt how much last night's sleep protects you from fatigue this evening?\n")
    
    def report_moderation(name, res, n_info):
        if not res: return f"  [{name}] Failed to fit."
        if 'Sleep_WP_z:Workload_WP_z' not in res:
            return f"  [{name}] Interaction term missing from fit (possible collinearity/aliasing)."
        sleep_main = res['Sleep_WP_z']['Estimate']
        interaction = res['Sleep_WP_z:Workload_WP_z']['Estimate']
        int_p = res['Sleep_WP_z:Workload_WP_z']['p_val']
        
        out = f"  [{name}] (N={int(n_info['n_obs_mean'])}, Participants={int(n_info['n_participants_mean'])})\n"
        out += f"    Sleep Main Effect:      Coef = {safe_format(sleep_main)}, p = {safe_format(res['Sleep_WP_z']['p_val'])}\n"
        out += f"    Interaction (SxW):      Coef = {safe_format(interaction)}, p = {safe_format(int_p)}\n"
        out += "    Conclusion: "
        if int_p < 0.05 and interaction > 0 and sleep_main < 0:
            out += "Workload blunts sleep's protective effect (supports hypothesis)."
        elif int_p < 0.05 and interaction < 0 and sleep_main < 0:
            out += "Sleep matters more under high workload (opposite direction)."
        else:
            out += "Interaction non-significant. No evidence of moderation (additive predictors only)."
        return out
        
    rep.append(report_moderation('Original (Complete Case)', orig_reml_mod, n_orig_mod))
    rep.append(report_moderation('Imputed (Passive, Pooled m=20)', imp_reml_mod_passive, n_imp_mod_passive))
    rep.append(report_moderation('Imputed (JAV, Pooled m=20)', imp_reml_mod_jav, n_imp_mod_jav))
    
    rep.append('\nTEST 2: NON-LINEAR DOSE-RESPONSE (Sleep + Sleep^2)')
    rep.append("Question: Is sleep's benefit linear, or does it show diminishing returns?\n")
    
    def report_dose(name, res, n_info):
        if not res: return f"  [{name}] Failed to fit."
        if 'I(Sleep_WP_z ** 2)' not in res:
            return f"  [{name}] Quadratic term missing from fit (possible collinearity/aliasing)."
        lin_coef = res['Sleep_WP_z']['Estimate']
        quad_coef = res['I(Sleep_WP_z ** 2)']['Estimate']
        quad_p = res['I(Sleep_WP_z ** 2)']['p_val']
        
        out = f"  [{name}] (N={int(n_info['n_obs_mean'])}, Participants={int(n_info['n_participants_mean'])})\n"
        out += f"    Linear (Sleep):         Coef = {safe_format(lin_coef)}, p = {safe_format(res['Sleep_WP_z']['p_val'])}\n"
        out += f"    Quadratic (Sleep^2):    Coef = {safe_format(quad_coef)}, p = {safe_format(quad_p)}\n"
        out += "    Conclusion: "
        if quad_p < 0.05 and quad_coef > 0 and lin_coef < 0:
            out += "Diminishing returns (supports hypothesis)."
        elif quad_p < 0.05 and quad_coef < 0 and lin_coef < 0:
            out += "Accelerating effect at extremes."
        else:
            out += "Quadratic non-significant. Linear model is adequate."
        return out
        
    rep.append(report_dose('Original (Complete Case)', orig_reml_dose, n_orig_dose))
    rep.append(report_dose('Imputed (Passive, Pooled m=20)', imp_reml_dose_passive, n_imp_dose_passive))
    rep.append(report_dose('Imputed (JAV, Pooled m=20)', imp_reml_dose_jav, n_imp_dose_jav))
    
    rt = '\n'.join(rep)
    print('\n' + rt)
    with open(REPORT_OUTPUT, 'w', encoding='utf-8') as f: f.write(rt)

if __name__ == '__main__':
    run_exploratory_sleep_models()
