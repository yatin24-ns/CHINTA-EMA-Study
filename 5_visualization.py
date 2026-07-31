"""PIPELINE STAGE 10: VISUALIZATION — read-only, saves to Plots/.
Plots 04-09 need enriched bayesian_linear_metrics.json (re-run 3_bayesian_linear.py if stale)."""

import os, json, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
import seaborn as sns
warnings.filterwarnings('ignore')
from stats_utils import BASE_DIR, DATASETS
from scipy.stats import gaussian_kde

PLOTS_DIR  = os.path.join(BASE_DIR, 'Plots');  os.makedirs(PLOTS_DIR, exist_ok=True)
FMT_DIR    = os.path.join(BASE_DIR, 'Formatted_Data')
ORIG_MISS  = os.path.join(FMT_DIR, 'original_with_missing.xlsx')
ORIG_COMP  = os.path.join(FMT_DIR, 'original_complete_cases.xlsx')
FL_JSON    = os.path.join(BASE_DIR, 'frequentist_linear_metrics.json')
BL_JSON    = os.path.join(BASE_DIR, 'bayesian_linear_metrics.json')
NC_ORIG    = os.path.join(BASE_DIR, 'bayesian_linear_2l_original.nc')
NC_IMP     = os.path.join(BASE_DIR, 'bayesian_linear_2l_imputed.nc')

SAVED, FAILED = [], []
SKIP = {'Intercept', 'Group Var', 'sigma', 'sigma_log__'}
PAL  = {'wl':'#E76F51','leis':'#2A9D8F','slp':'#6A4C93','fat':'#F4A261','hi':'#264653','neu':'#A8DADC'}
MS   = {  # MODEL_STYLES — centralized, used identically in all comparison plots
    'Freq (Original)':  {'color':'#264653','marker':'o'},
    'Freq (Imputed)':   {'color':'#2A9D8F','marker':'s'},
    'Bayes (Original)': {'color':'#E76F51','marker':'^'},
    'Bayes (Imputed)':  {'color':'#E9C46A','marker':'D'},
}
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':0.3,'figure.dpi':150})

# ── helpers ───────────────────────────────────────────────────────────────
def _save(fig, name):
    fig.savefig(os.path.join(PLOTS_DIR, name), bbox_inches='tight')
    plt.close(fig); SAVED.append(name); print(f'  [OK] {name}')

def _fail(name, e):
    FAILED.append((name, str(e))); print(f'  [FAIL] {name}: {e}')

def _sc(v):   # extract scalar from possible dict
    return v.get('mean', np.nan) if isinstance(v, dict) else (v if v is not None else np.nan)

def _require_enriched(bl):
    if not any('_hdi_lb' in k for k in bl.get('Linear_2L_imputed', {})):
        raise RuntimeError("Enriched Bayesian metrics missing — re-run 3_bayesian_linear.py")

def _load_metrics():
    fl = json.load(open(FL_JSON)); bl = json.load(open(BL_JSON))
    _require_enriched(bl)
    m = {}
    for key, label in [('Linear_2L_original','Freq (Original)'),('Linear_2L_imputed','Freq (Imputed)')]:
        m[label] = {p: {'est':v['Estimate'],'se':v['SE'],
                        'hdi_lb':v['Estimate']-1.96*v['SE'],'hdi_ub':v['Estimate']+1.96*v['SE'],
                        'p_val':v.get('p_val',np.nan),'pd':np.nan,'interval_type':'95% CI'}
                    for p,v in fl.get(key,{}).get('Parameters',{}).items() if p not in SKIP}
    for key, label in [('Linear_2L_original','Bayes (Original)'),('Linear_2L_imputed','Bayes (Imputed)')]:
        blk = bl.get(key, {})
        params = {k[5:-5] for k in blk if k.startswith('post_') and k.endswith('_mean')}
        m[label] = {p: {'est':blk.get(f'post_{p}_mean',np.nan),'se':blk.get(f'post_{p}_sd',np.nan),
                         'hdi_lb':blk.get(f'post_{p}_hdi_lb',np.nan),'hdi_ub':blk.get(f'post_{p}_hdi_ub',np.nan),
                         'pd':blk.get(f'post_{p}_pd',np.nan),'p_val':np.nan,'interval_type':'95% HDI',
                         'rhat_per_imp':blk.get(f'post_{p}_rhat_per_imp',[])}
                    for p in params if p not in SKIP}
    return m, fl, bl

# ── retained plots ────────────────────────────────────────────────────────
def plot_missingness_heatmap():
    df = pd.read_excel(ORIG_MISS)
    vmap = {'Workload':'S1_Workload_Mean','Fatigue':'S2_Fatigue_Mean','Leisure':'S3_Leisure_Mean','Sleep':'Sleep_Quality'}
    ps = sorted(df['Participant_No'].unique()); ds = sorted(df['day_number'].unique())
    cmap = matplotlib.colors.ListedColormap(['#E63946','#A8DADC'])
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Missingness Heatmap: Participant × Day', fontsize=14, fontweight='bold', y=1.01)
    for ax, (label, col) in zip(axes.flat, vmap.items()):
        mx = pd.DataFrame(index=ps, columns=ds, dtype=float)
        for _, r in df.iterrows(): mx.loc[r['Participant_No'], r['day_number']] = 0 if pd.isna(r[col]) else 1
        sns.heatmap(mx.astype(float), ax=ax, cmap=cmap, cbar=True, cbar_kws={'ticks':[.25,.75],'shrink':.6},
                    linewidths=.3, linecolor='#EEE', vmin=0, vmax=1)
        ax.set_title(f'{label} ({col})', fontweight='bold', fontsize=10)
        ax.set_xlabel('Study Day'); ax.set_ylabel('Participant')
        ax.collections[0].colorbar.set_ticklabels(['Missing','Observed'])
    plt.tight_layout(); _save(fig, '01_missingness_heatmap.png')

def plot_imputation_density_overlay():
    df0 = pd.read_excel(ORIG_MISS); di = pd.read_parquet(DATASETS['imputed_passive'])
    cfg = [('S1_Workload_Mean','Workload',PAL['wl']),('S3_Leisure_Mean','Leisure',PAL['leis']),('Sleep_Quality','Sleep',PAL['slp'])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Imputation Density Overlay: Observed vs. Imputed (m=20)', fontsize=13, fontweight='bold')
    for ax, (col, label, color) in zip(axes, cfg):
        obs = df0[col].dropna()
        for imp in sorted(di['.imp'].unique()):
            di[di['.imp']==imp][col].dropna().plot.kde(ax=ax, color=color, alpha=.08, lw=.8)
        obs.plot.kde(ax=ax, color=PAL['hi'], lw=2.5, label='Observed')
        ax.legend(handles=[ax.lines[0], mpatches.Patch(color=color, alpha=.5, label='Imputed (m=20)')],
                  labels=['Observed','Imputed (m=20)'], fontsize=8)
        ax.set_title(label, fontweight='bold'); ax.set_xlabel('Value'); ax.set_ylabel('Density')
    plt.tight_layout(); _save(fig, '02_imputation_density_overlay.png')

def plot_spaghetti():
    df = pd.read_excel(ORIG_COMP)
    fig, ax = plt.subplots(figsize=(14, 6))
    for pid in df['Participant_No'].unique():
        p = df[df['Participant_No']==pid].sort_values('day_number')
        ax.plot(p['day_number'], p['S2_Fatigue_Mean_z'], color=PAL['fat'], alpha=.25, lw=1.)
    gm = df.groupby('day_number')['S2_Fatigue_Mean_z'].mean()
    ax.plot(gm.index, gm.values, color=PAL['hi'], lw=2.5, label='Group Mean')
    ax.axhline(0, color='grey', lw=.8, ls='--'); ax.set_xlabel('Study Day')
    ax.set_ylabel('Fatigue (z)'); ax.set_title('Within-Person Fatigue Trajectories', fontweight='bold', fontsize=13)
    ax.legend(); _save(fig, '03_spaghetti_fatigue_trajectories.png')

# ── new comparison plots ──────────────────────────────────────────────────
def plot_forest():
    m, _, _ = _load_metrics()
    all_p = sorted({p for d in m.values() for p in d})
    wp = [p for p in all_p if any(x in p for x in ('WP','day','weekend'))]
    ordered = wp + [p for p in all_p if p not in wp]
    yp = {p:i for i,p in enumerate(ordered)}
    offsets = np.linspace(-.28, .28, 4)
    fig, ax = plt.subplots(figsize=(11, max(7, len(ordered)*.55)))
    for (lbl, style), off in zip(MS.items(), offsets):
        for p, v in m.get(lbl, {}).items():
            y = yp.get(p);
            if y is None: continue
            ax.errorbar(v['est'], y+off, xerr=[[v['est']-v['hdi_lb']],[v['hdi_ub']-v['est']]],
                        fmt='none', color=style['color'], alpha=.6, capsize=3, lw=1.2)
            ax.scatter(v['est'], y+off, color=style['color'], marker=style['marker'], s=50, zorder=4)
    ax.axvline(0, color='black', lw=1., ls='--', alpha=.5)
    ax.set_yticks(list(yp.values())); ax.set_yticklabels([p.replace('_',' ') for p in ordered], fontsize=8.5)
    ax.set_xlabel('Estimate  (Freq: ±95% CI  |  Bayes: ±95% HDI)', fontsize=10)
    ax.set_title('Coefficient Forest — 4 Variants', fontweight='bold', fontsize=12)
    ax.legend(handles=[mpatches.Patch(color=MS[l]['color'], label=l) for l in MS], fontsize=8, loc='lower right')
    plt.tight_layout(); _save(fig, '04_coefficient_forest.png')

def plot_freq_vs_bayes():
    m, _, _ = _load_metrics()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('Frequentist vs. Bayesian Agreement\n(x-err: ±95%CI  y-err: ±posterior SD)', fontsize=12, fontweight='bold')
    for ax, (fl, bl, title) in zip(axes, [('Freq (Original)','Bayes (Original)','Original'),
                                           ('Freq (Imputed)', 'Bayes (Imputed)', 'Imputed')]):
        fd, bd = m[fl], m[bl]; common = [p for p in fd if p in bd]
        xs, ys = [fd[p]['est'] for p in common], [bd[p]['est'] for p in common]
        ax.errorbar(xs, ys, xerr=[fd[p]['se']*1.96 for p in common], yerr=[bd[p]['se'] for p in common],
                    fmt='none', color=PAL['neu'], alpha=.7, capsize=3, lw=1.)
        ax.scatter(xs, ys, color=MS[fl]['color'], s=60, zorder=3)
        lim = max(abs(np.array(xs+ys)))*1.2; ax.plot([-lim,lim],[-lim,lim],'--',color='grey',lw=1.)
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Freq Estimate'); ax.set_ylabel('Bayes Posterior Mean')
        for p, x, y in zip(common, xs, ys):
            if abs(x-y) > 0.05: ax.annotate(p.replace('_',' '),(x,y),xytext=(5,3),textcoords='offset points',fontsize=7)
    plt.tight_layout(); _save(fig, '05_freq_vs_bayes_scatter.png')

def plot_orig_vs_imputed():
    m, _, _ = _load_metrics()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('Original vs. Imputed Estimates\n(points far from diagonal = imputation-sensitive)', fontsize=12, fontweight='bold')
    for ax, (ol, il, title) in zip(axes, [('Freq (Original)','Freq (Imputed)','Frequentist'),
                                           ('Bayes (Original)','Bayes (Imputed)','Bayesian')]):
        od, id_ = m[ol], m[il]; common = [p for p in od if p in id_]
        xs, ys = [od[p]['est'] for p in common], [id_[p]['est'] for p in common]
        dists = [abs(x-y) for x,y in zip(xs,ys)]; thresh = np.percentile(dists, 70)
        ax.scatter(xs, ys, color=MS[ol]['color'], s=60, zorder=3)
        lim = max(abs(np.array(xs+ys)))*1.2; ax.plot([-lim,lim],[-lim,lim],'--',color='grey',lw=1.)
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Complete-Cases Estimate'); ax.set_ylabel('Imputed Estimate')
        for p, x, y in zip(common, xs, ys):
            ax.annotate(p.replace('_',' '),(x,y),xytext=(5,3),textcoords='offset points',fontsize=7)
    plt.tight_layout(); _save(fig, '06_orig_vs_imputed_scatter.png')

def plot_posterior_densities():
    try: import xarray as xr, arviz as az
    except ImportError: raise RuntimeError("arviz/xarray not available")
    found = {k:v for k,v in {'Original':NC_ORIG,'Imputed':NC_IMP}.items() if os.path.exists(v)}
    if not found: raise FileNotFoundError("No .nc traces found — run Stage 3 first")
    idatas = {n: xr.open_datatree(p) for n,p in found.items()}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Posterior Distributions — Key Workload Effects', fontsize=13, fontweight='bold')
    colors = {'Original':PAL['wl'],'Imputed':PAL['hi']}
    for ax, param in zip(axes, ['Workload_WP_z','Workload_BP_z']):
        for name, idata in idatas.items():
            try:
                draws = idata['/posterior'].ds[param].values.flatten(); c = colors[name]
                xr_ = np.linspace(draws.min(), draws.max(), 300); kde = gaussian_kde(draws)
                ax.plot(xr_, kde(xr_), color=c, lw=2, label=name)
                lb, ub = np.percentile(draws, [2.5, 97.5]); mask = (xr_>=lb)&(xr_<=ub)
                ax.fill_between(xr_[mask], kde(xr_)[mask], alpha=.2, color=c)
            except Exception: pass
        ax.axvline(0, color='black', lw=1.2, ls='--', alpha=.7)
        ax.set_title(param.replace('_',' '), fontweight='bold'); ax.set_xlabel('Coefficient'); ax.set_ylabel('Density')
        ax.legend(fontsize=9, title='95% HDI shaded')
    plt.tight_layout(); _save(fig, '07_posterior_densities.png')

def plot_pd_bars():
    m, _, _ = _load_metrics()
    freq, bayes = m['Freq (Imputed)'], m['Bayes (Imputed)']
    common = [p for p in freq if p in bayes and not np.isnan(bayes[p]['pd'])]
    x = np.arange(len(common)); w = .38
    fig, ax = plt.subplots(figsize=(max(10, len(common)*.9), 6))
    ax.bar(x-w/2, [1-freq[p]['p_val'] for p in common], w, label='1−p  (Freq)',  color=MS['Freq (Imputed)']['color'],  alpha=.85)
    ax.bar(x+w/2, [bayes[p]['pd']     for p in common], w, label='pd  (Bayes)',  color=MS['Bayes (Imputed)']['color'], alpha=.85)
    ax.axhline(.95, color='black', lw=.9, ls='--', alpha=.5, label='0.95 threshold')
    ax.set_xticks(x); ax.set_xticklabels([p.replace('_',' ') for p in common], rotation=35, ha='right', fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Evidence strength (both → 1 as evidence increases)')
    ax.set_title('Cross-Paradigm Evidence Strength (Imputed)\n1−p and pd: higher = stronger', fontweight='bold')
    ax.legend(fontsize=9); plt.tight_layout(); _save(fig, '08_pd_pvalue_bars.png')

def plot_rhat():
    _, _, bl = _load_metrics(); _require_enriched(bl)
    blk = bl.get('Linear_2L_imputed', {})
    params, per_imp, pooled = [], [], []
    pooled_map = {}
    sp = os.path.join(BASE_DIR, 'bayesian_linear_2l_imputed_summary.txt')
    if os.path.exists(sp):
        import re
        for line in open(sp):
            mt = re.match(r'\s*(\S+)\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+([\d.]+)', line)
            if mt: pooled_map[mt.group(1)] = float(mt.group(2))
    for k, v in blk.items():
        if not (k.startswith('post_') and k.endswith('_rhat_per_imp') and v): continue
        p = k[5:-13]
        if p in SKIP: continue
        params.append(p); per_imp.append(v); pooled.append(pooled_map.get(p, np.nan))
    if not params: raise RuntimeError("No rhat_per_imp data — re-run 3_bayesian_linear.py")
    fig, ax = plt.subplots(figsize=(max(10, len(params)*.8), 6))
    ax.boxplot(per_imp, patch_artist=True, boxprops=dict(facecolor=PAL['neu'],alpha=.7), medianprops=dict(color=PAL['hi'],lw=2))
    for i, r in enumerate(pooled, 1):
        if not np.isnan(r): ax.scatter(i, r, color='#E63946', s=60, zorder=5, marker='D')
    ax.axhline(1.05, color='#E63946', lw=1.2, ls='--', label='R-hat = 1.05')
    ax.axhline(1.00, color='grey',    lw=.8,  ls=':',  alpha=.5)
    ax.set_xticks(range(1, len(params)+1)); ax.set_xticklabels([p.replace('_',' ') for p in params], rotation=35, ha='right', fontsize=8.5)
    ax.set_ylabel('R-hat')
    ax.set_title('Convergence Diagnostics — Imputed Model Only\nBoxes = per-imputation R-hat | ◆ = pooled (concatenation artifact if elevated)', fontweight='bold', fontsize=10)
    ax.legend(handles=[mpatches.Patch(facecolor=PAL['neu'],alpha=.7,label='Per-imp (box)'),
                        mpatches.Patch(facecolor='#E63946',label='Pooled (◆)')], fontsize=8)
    plt.tight_layout(); _save(fig, '09_rhat_diagnostics.png')

def plot_model_fit():
    fl = json.load(open(FL_JSON)); bl = json.load(open(BL_JSON))
    ao, bo, ai, bi = _sc(fl.get('Linear_2L_original',{}).get('AIC_ML')), _sc(fl.get('Linear_2L_original',{}).get('BIC_ML')), \
                     _sc(fl.get('Linear_2L_imputed', {}).get('AIC_ML')), _sc(fl.get('Linear_2L_imputed', {}).get('BIC_ML'))
    lo, li = _sc(bl.get('Linear_2L_original',{}).get('LOO_elpd_mean')), _sc(bl.get('Linear_2L_imputed',{}).get('LOO_elpd_mean'))
    lo_sd, li_sd = _sc(bl.get('Linear_2L_original',{}).get('LOO_elpd_sd')) or 0, _sc(bl.get('Linear_2L_imputed',{}).get('LOO_elpd_sd')) or 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Model Fit Comparison', fontsize=13, fontweight='bold')
    x = np.arange(2)
    ax1.bar(x-.2,[ao,ai],.35,label='AIC',color=[MS['Freq (Original)']['color'],MS['Freq (Imputed)']['color']],alpha=.85)
    ax1.bar(x+.2,[bo,bi],.35,label='BIC',color=[MS['Freq (Original)']['color'],MS['Freq (Imputed)']['color']],alpha=.5)
    ax1.set_xticks(x); ax1.set_xticklabels(['Original\n(complete cases)','Imputed\n(m=20, larger N)'])
    ax1.set_ylabel('AIC / BIC'); ax1.legend(fontsize=8); ax1.set_title('Frequentist: AIC & BIC', fontweight='bold')
    ax1.text(.5,.97,'⚠ N differs — not directly comparable',transform=ax1.transAxes,ha='center',va='top',fontsize=8,color='#E63946',style='italic')
    ax2.bar([0,1],[lo,li],.5,color=[MS['Bayes (Original)']['color'],MS['Bayes (Imputed)']['color']],alpha=.85,yerr=[lo_sd,li_sd],capsize=5)
    ax2.set_xticks([0,1]); ax2.set_xticklabels(['Bayes Original','Bayes Imputed'])
    ax2.set_ylabel('LOO ELPD (higher = better)'); ax2.set_title('Bayesian: LOO-ELPD', fontweight='bold')
    plt.tight_layout(); _save(fig, '10_model_fit_comparison.png')

# ── spot checks ───────────────────────────────────────────────────────────
def _spot_checks():
    try:
        fl=json.load(open(FL_JSON)); bl=json.load(open(BL_JSON)); m,_,_=_load_metrics(); ok=0
        def ck(a,b,n): 
            nonlocal ok
            assert abs(a-b)<1e-6, f"Check {n} FAIL: {a} vs {b}"; ok+=1
        ck(fl['Linear_2L_imputed']['Parameters']['Workload_WP_z']['Estimate'], m['Freq (Imputed)']['Workload_WP_z']['est'], 1)
        lb = bl['Linear_2L_imputed'].get('post_Workload_WP_z_hdi_lb')
        if lb: ck(lb, m['Bayes (Imputed)']['Workload_WP_z']['hdi_lb'], 2)
        pd_ = bl['Linear_2L_imputed'].get('post_Workload_WP_z_pd')
        if pd_: ck(pd_, m['Bayes (Imputed)']['Workload_WP_z']['pd'], 3)
        rl = bl['Linear_2L_imputed'].get('post_Workload_WP_z_rhat_per_imp')
        if rl: assert isinstance(rl,list) and all(isinstance(r,float) for r in rl); ok+=1
        bp = bl['Linear_2L_original'].get('post_Workload_BP_z_mean')
        if bp: ck(bp, m['Bayes (Original)'].get('Workload_BP_z',{}).get('est',bp), 5)
        print(f'\n  SPOT CHECKS: {ok}/5 PASSED')
    except Exception as e: print(f'\n  SPOT CHECKS: {e}')

# ── appendix (excluded from default run, call manually) ───────────────────
def _appendix(): pass  # Original demoted plots kept here for on-demand use

# ── main ──────────────────────────────────────────────────────────────────
PLOTS = [
    ('01 Missingness Heatmap',       plot_missingness_heatmap),
    ('02 Imputation Density Overlay',plot_imputation_density_overlay),
    ('03 Spaghetti Trajectories',    plot_spaghetti),
    ('04 Coefficient Forest',        plot_forest),
    ('05 Freq vs Bayes Scatter',     plot_freq_vs_bayes),
    ('06 Orig vs Imputed Scatter',   plot_orig_vs_imputed),
    ('07 Posterior Densities',       plot_posterior_densities),
    ('08 pd / p-value Bars',         plot_pd_bars),
    ('09 R-hat Diagnostics',         plot_rhat),
    ('10 Model Fit Comparison',      plot_model_fit),
]

def generate_all_plots():
    print('='*70); print(f'  STAGE 10: VISUALIZATION  |  {PLOTS_DIR}'); print('='*70)
    for name, fn in PLOTS:
        print(f'\n  {name}')
        try: fn()
        except Exception as e: _fail(name, e)
    _spot_checks()
    print(f'\n{"="*70}\n  COMPLETE — {len(SAVED)} saved, {len(FAILED)} failed.')
    if SAVED:  [print(f'    ok {p}') for p in SAVED]
    if FAILED: [print(f'    not ok {n}: {e}') for n,e in FAILED]

if __name__ == '__main__': generate_all_plots()
