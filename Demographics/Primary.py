import pandas as pd, hashlib, numpy as np, seaborn as sns, tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt, matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid") 
plt.rcParams.update({
    'font.family': 'sans-serif', 'axes.edgecolor': '#cccccc', 'axes.linewidth': 0.8,
    'grid.color': '#eeeeee', 'grid.linewidth': 0.5, 'figure.titlesize': 14,
    'figure.titleweight': 'bold', 'figure.facecolor': '#f8f9fa', 'axes.facecolor': '#ffffff'
})

BASE = r"C:\Users\praka\OneDrive\Desktop\CHINTA practice\Demographics"
INPUT, OUTPUT = fr"{BASE}\EMA consent form and Trait-behaviour (Responses).xlsx", fr"{BASE}\output.xlsx"
FIG_OVERVIEW, FIG_ADVANCED = fr"{BASE}\fig1_overview.png", fr"{BASE}\fig2_advanced.png"
FIG_DEMO, FIG_FATIGUE = fr"{BASE}\fig3_demographics.png", fr"{BASE}\fig4_fatigue.png"

# Extra whitespace added around every saved figure so titles/labels are never clipped
SAVE_KW = dict(dpi=200, bbox_inches='tight', pad_inches=0.6)

LIKERT = {"doesn't apply at all": 1, "does not apply much": 2, "slightly applies": 3, "applies a lot": 4, "applies completely": 5}
EDU_RANK = {"no formal education": 1, "high school diploma or equivalent (10th/12th)": 2, "associate degree or vocational training": 3,
            "bachelor's degree": 4, "master's degree": 5, "doctorate or professional degree (e.g., phd, md, jd)": 6}
AGE_BINS, AGE_LABELS = [0, 20, 25, 30, 35, 40, 50, 100], ['<=20', '21-25', '26-30', '31-35', '36-40', '41-50', '50+']
PAL = sns.color_palette("deep")

def load(path):
    df = pd.read_excel(path) 
    ok = lambda c: df.iloc[:, c].astype(str).str.strip().str.lower().str.startswith('yes')
    df = df[ok(1) & ok(2)].reset_index(drop=True) #Confirming consent of the participant- "Yes"
    return df if len(df) else None

def process(path):
    df = load(path)
    if df is None: return None
    age_c, gen_c, edu_c = df.columns[5], df.columns[6], df.columns[7]
    lik_c = list(df.columns[8:18])

    df.insert(0, 'P_No', [f"P{i:03d}" for i in range(1, len(df) + 1)]) #SHA-256 hash and convert to 8-digit integer
    df.insert(1, 'P_ID', df[age_c].apply(lambda x: int(hashlib.sha256(str(x).strip().lower().encode()).hexdigest(), 16) % 10**8 if pd.notna(x) else None))
    #Reordering
    df[age_c] = pd.to_numeric(df[age_c], errors='coerce')
    df['Age_Group'] = pd.cut(df[age_c], AGE_BINS, labels=AGE_LABELS, right=True)
    df['Gender'] = df[gen_c].apply(lambda r: {'m': 'Male', 'f': 'Female'}.get(str(r).strip().lower()[0], 'Other') if pd.notna(r) else 'Other')
    df['Education'] = df[edu_c].astype(str).str.strip().str.lower()
    df['Edu_Rank'] = df['Education'].map(EDU_RANK)

    qn = [f"Q{i}" for i in range(1, 11)]
    df[qn] = df[lik_c].astype(str).apply(lambda c: c.str.strip().str.lower().map(LIKERT))
    df['Fat_Total'] = df[qn].sum(axis=1)
    df['Fat_Mean'], df['Fat_SD'] = df[qn].mean(axis=1), df[qn].std(axis=1)
    df['Extreme_5s'] = (df[qn] == 5).sum(axis=1)
    cohort_mean, cohort_sd = df['Fat_Total'].mean(), df['Fat_Total'].std()
    df['High_Risk'] = df['Fat_Total'] >= (cohort_mean + cohort_sd)

    freq = lambda col: df[col].value_counts().reset_index().rename(columns={'count': 'N'}).assign(pct=lambda x: (x['N'] / x['N'].sum() * 100).round(1))
    
    rows = [
        {'Factor': fac, 'Level': str(lv).title() if fac == 'Education' else str(lv), 'N': len(s),
         'Mean': round(s.mean(), 2), 'SD': round(s.std(), 2), 'Median': round(s.median(), 2),
         'High_Risk_N': int((s >= cohort_mean + cohort_sd).sum())}
        for fac, col in [('Gender', 'Gender'), ('Age Group', 'Age_Group'), ('Education', 'Education')]
        for lv, s in df.groupby(col, observed=False)['Fat_Total']
    ]

    q1, q3 = df[age_c].quantile(.25), df[age_c].quantile(.75)
    tables = {
        'age_stats': pd.DataFrame({'Metric': ['Mean', 'Median', 'SD', 'Min', 'Max', 'Q1', 'Q3', 'IQR', 'Skew'],
                                    'Value': [df[age_c].mean(), df[age_c].median(), df[age_c].std(), df[age_c].min(), df[age_c].max(), q1, q3, q3 - q1, df[age_c].skew()]}).round(2),
        'age_dist': freq('Age_Group').sort_values('Age_Group').reset_index(drop=True),
        'gen_dist': freq('Gender'),
        'gen_age': pd.crosstab(df['Gender'], df['Age_Group']),
        'edu_dist': freq('Education'),
        'fat_stats': pd.DataFrame({'Metric': ['Cohort Mean', 'SD', 'Median', 'Min', 'Max', 'High-Risk Count'],
                                    'Value': [round(cohort_mean, 2), round(cohort_sd, 2), round(df['Fat_Total'].median(), 2), df['Fat_Total'].min(), df['Fat_Total'].max(), int(df['High_Risk'].sum())]}),
        'item_means': df[qn].mean().round(2).to_frame('Mean'),
        'combined': pd.DataFrame(rows),
    }
    return df, tables, age_c, qn

#The dedicated library for making rainclouds is called- "ptitprince"
#I am too lazy to learn another language just for rainclouds
def draw_raincloud(data, x_col, y_col, ax, palette, title, ylabel, xlabel, squeeze=False, orient='v'):
    cats = data[x_col].dropna().unique()
    if x_col == 'Age_Group': cats = sorted(cats, key=str)
    vw, bw = (0.4, 0.08) if squeeze else (0.6, 0.15)
    horiz = orient != 'v'

    for i, cat in enumerate(cats):
        sub = data[data[x_col] == cat][y_col].dropna().values
        if len(sub) == 0: continue
        color = palette[i % len(palette)]
        v = ax.violinplot(sub, positions=[i], vert=not horiz, showmeans=False, showmedians=False, showextrema=False, widths=vw)
        for b in v['bodies']:
            verts = b.get_paths()[0].vertices
            verts[:, 1 if horiz else 0] = np.clip(verts[:, 1 if horiz else 0], i, np.inf)
            b.set(facecolor=color, edgecolor='black', alpha=0.6, linewidth=0.5)

        bp = ax.boxplot(sub, positions=[i - 0.15], vert=not horiz, widths=bw, showfliers=False, patch_artist=True)
        for box in bp['boxes']: box.set(facecolor='white', alpha=0.8)
        for median in bp['medians']: median.set(color='black', linewidth=1.5)

        jitter = np.random.uniform(i - 0.4, i - 0.2, size=len(sub))
        scatter_args = (sub, jitter) if horiz else (jitter, sub)
        ax.scatter(*scatter_args, s=10, color=color, alpha=0.6, zorder=2, edgecolors='black', linewidths=0.5)

    labels = [str(c).title()[:15] + '..' if len(str(c)) > 15 else str(c).title() for c in cats]
    if horiz:
        ax.set_yticks(range(len(cats))); ax.set_yticklabels(labels)
        ax.set(xlabel=ylabel, ylabel=xlabel)
    else:
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(labels)
        ax.set(ylabel=ylabel, xlabel=xlabel)
    ax.set_title(title, weight='bold')

def draw_3d_surface(df, age_col, ax_3d, title):
    ax_3d.set_facecolor('#f8f9fa')
    sub = df.dropna(subset=[age_col, 'Edu_Rank', 'Fat_Total'])
    for g, color in zip(['Male', 'Female'], ['#3498db', '#e74c3c']):
        g_sub = sub[sub['Gender'] == g]
        if len(g_sub) >= 3:
            X_vals, Y_vals = g_sub[[age_col, 'Edu_Rank']].values, g_sub['Fat_Total'].values
            A_mat = np.c_[X_vals[:, 0], X_vals[:, 1], np.ones(X_vals.shape[0])]
            C, *_ = np.linalg.lstsq(A_mat, Y_vals, rcond=None)
            x_surf, y_surf = np.meshgrid(np.linspace(X_vals[:, 0].min(), X_vals[:, 0].max(), 10), np.linspace(X_vals[:, 1].min(), X_vals[:, 1].max(), 10))
            z_surf = C[0] * x_surf + C[1] * y_surf + C[2]
            ax_3d.plot_surface(x_surf, y_surf, z_surf, color=color, alpha=0.6, edgecolor='none')
            ax_3d.scatter(g_sub[age_col], g_sub['Edu_Rank'], g_sub['Fat_Total'], color=color, label=g, s=25, alpha=0.9, edgecolors='white')
    ax_3d.set(xlabel='Age (yrs)', ylabel='Edu Rank (1-6)', zlabel='Fatigue Total')
    ax_3d.set_title(title, weight='bold')
    ax_3d.legend(fontsize=9, loc='upper left')

def plot_3d_standalone(df, age_col): 
    #I have kept the 3d plot separate as it can be displayed in a separate window
    fig = plt.figure(figsize=(10, 8))
    ax_3d = fig.add_subplot(111, projection='3d')
    draw_3d_surface(df, age_col, ax_3d, title='3D Demographic Surface by Gender')
    fig.tight_layout(pad=2.0)
    return fig

def _hist_strip(df, col, color, strip_color, ax, title, xlabel):
    sns.histplot(data=df, x=col, bins=20, kde=True, color=color, ax=ax, edgecolor='white')
    ax2 = ax.twinx()
    sns.stripplot(data=df, x=col, color=strip_color, ax=ax2, size=4, alpha=0.5, jitter=0.1)
    ax2.set_ylim(-1, 5); ax2.set_yticks([]); sns.despine(ax=ax2, right=True, left=True)
    mean_val = df[col].mean()
    ax.axvline(mean_val, color='red', ls='--', label=f'Mean={mean_val:.1f}')
    ax.legend(fontsize=8)
    ax.set(title=title, xlabel=xlabel, ylabel='Count (N)')

#I am too lazy to use a new library for raincouds- the dedicated package for that purpose is called "ptitprince" if anyone finds that more convenient
def plot_demographics_2x2(df, age_col):
    fig, ax = plt.subplots(2, 2, figsize=(14, 12), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    fig.suptitle('Cohort Demographics', fontsize=16, weight='bold', y=0.96)

    _hist_strip(df, age_col, PAL[0], 'darkorange', ax[0, 0], 'A. Age Distribution', 'Age (yrs)')
    draw_raincloud(df, 'Gender', age_col, ax[0, 1], sns.color_palette("Set2"), 'B. Age by Gender', 'Age', 'Gender')
    draw_raincloud(df, 'Education', age_col, ax[1, 0], sns.color_palette("deep"), 'C. Age by Education', 'Age', 'Education Level', orient='h')
    _hist_strip(df, 'Fat_Total', PAL[3], 'magenta', ax[1, 1], 'D. Fatigue Score Distribution', 'Total Fatigue Score (10-50)')

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94], pad=2.0)
    plt.savefig(FIG_DEMO, **SAVE_KW)
    return fig

def plot_fatigue_2x2(df, qcols):
    fig, ax = plt.subplots(2, 2, figsize=(14, 12), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    fig.suptitle('Fatigue Across Demographics', fontsize=16, weight='bold', y=0.96)

    draw_raincloud(df, 'Gender', 'Fat_Total', ax[0, 0], sns.color_palette("Set2"), 'A. Fatigue by Gender', 'Fatigue Score', 'Gender', squeeze=True)
    draw_raincloud(df, 'Age_Group', 'Fat_Total', ax[0, 1], sns.color_palette("Set1"), 'B. Fatigue by Age Group', 'Fatigue Score', 'Age Group')
    draw_raincloud(df, 'Education', 'Fat_Total', ax[1, 0], sns.color_palette("Set3"), 'C. Fatigue by Education', 'Fatigue Score', 'Education Level', orient='h')

    means = df[qcols].mean()
    norm = plt.Normalize(means.min(), means.max())
    colors = [cm.get_cmap('coolwarm')(norm(v)) for v in means.values]
    sns.barplot(x=list(range(1, 11)), y=means.values, palette=colors, ax=ax[1, 1], legend=False, edgecolor='black')
    for i, v in enumerate(means.values): ax[1, 1].text(i, v + 0.1, f"{v:.2f}", ha='center', va='bottom', fontsize=8, weight='bold')
    ax[1, 1].axhline(3, color='gray', ls='--', alpha=.5, label='Neutral (3.0)')
    ax[1, 1].set(xlabel='Item #', ylabel='Mean Score (1-5)', xticks=range(0, 10), xticklabels=range(1, 11), ylim=(0, 6.0))
    ax[1, 1].set_title('D. Per-Item Means', weight='bold')
    ax[1, 1].legend(fontsize=7, loc='upper left')

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94], pad=2.0)
    plt.savefig(FIG_FATIGUE, **SAVE_KW)
    return fig

#This will be displayed when you click on its assigned tab on the output window
def make_table_view(parent, title, tbl):
    frame = ttk.LabelFrame(parent, text=title, padding=10)
    text = tk.Text(frame, wrap=tk.NONE, height=min(len(tbl) + 3, 15), font=("Consolas", 10), bd=0, highlightthickness=0)
    show_idx = tbl.index.name is not None or not isinstance(tbl.index, pd.RangeIndex) or any(k in title.lower() for k in ['crosstab', 'item', 'stats'])
    text.insert(tk.END, tbl.to_string(index=show_idx))
    text.configure(state=tk.DISABLED)
    xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=text.xview)
    text.configure(xscrollcommand=xscroll.set)
    text.pack(fill=tk.BOTH, expand=True); xscroll.pack(fill=tk.X)
    return frame

    
def plot_overview(df, age_col, qcols):
    fig, ax = plt.subplots(2, 4, figsize=(28, 13), gridspec_kw={'hspace': 0.5, 'wspace': 0.35})
    fig.suptitle('Demographic & Trait Fatigue Overview', fontsize=18, weight='bold', y=0.98)

    _hist_strip(df, age_col, PAL[0], 'darkorange', ax[0, 0], 'A. Age Distribution', 'Age (yrs)')
    draw_raincloud(df, 'Gender', age_col, ax[0, 1], sns.color_palette("Set2"), 'B. Age by Gender', 'Age', 'Gender')
    draw_raincloud(df, 'Education', age_col, ax[0, 2], sns.color_palette("deep"), 'C. Age by Education', 'Age', 'Education Level', orient='h')
    _hist_strip(df, 'Fat_Total', PAL[3], 'magenta', ax[0, 3], 'D. Fatigue Score Distribution', 'Total Fatigue Score (10-50)')

    draw_raincloud(df, 'Gender', 'Fat_Total', ax[1, 0], sns.color_palette("Set2"), 'E. Fatigue by Gender', 'Fatigue Score', 'Gender', squeeze=True)
    draw_raincloud(df, 'Age_Group', 'Fat_Total', ax[1, 1], sns.color_palette("Set1"), 'F. Fatigue by Age Group', 'Fatigue Score', 'Age Group')
    draw_raincloud(df, 'Education', 'Fat_Total', ax[1, 2], sns.color_palette("Set3"), 'G. Fatigue by Education', 'Fatigue Score', 'Education Level', orient='h')

    means = df[qcols].mean()
    norm = plt.Normalize(means.min(), means.max())
    colors = [cm.get_cmap('coolwarm')(norm(v)) for v in means.values]
    sns.barplot(x=list(range(1, 11)), y=means.values, palette=colors, ax=ax[1, 3], legend=False, edgecolor='black')
    for i, v in enumerate(means.values): ax[1, 3].text(i, v + 0.1, f"{v:.2f}", ha='center', va='bottom', fontsize=8, weight='bold')
    ax[1, 3].axhline(3, color='gray', ls='--', alpha=.5, label='Neutral (3.0)')
    ax[1, 3].set(xlabel='Item #', ylabel='Mean Score (1-5)', xticks=range(0, 10), xticklabels=range(1, 11), ylim=(0, 6.0))
    ax[1, 3].set_title('H. Per-Item Means', weight='bold')
    ax[1, 3].legend(fontsize=7, loc='upper left')

    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.93], pad=3.0)
    plt.savefig(FIG_OVERVIEW, **SAVE_KW)
    return fig


def plot_advanced_analytics(df, age_col, qcols):
    fig = plt.figure(figsize=(18, 15))
    fig.suptitle('Extended Fatigue Analysis', fontsize=16, weight='bold', y=0.995)
    # top=0.90 reserves a clear gap between the suptitle and the first row of panels, this is something i have spent a unreasonable amont of time on 
    gs = fig.add_gridspec(6, 6, hspace=0.9, wspace=0.9, top=0.90, bottom=0.05)

    # A. Joint scatter + marginals
    ax_scatter = fig.add_subplot(gs[1:3, 0:2])
    ax_x, ax_y = fig.add_subplot(gs[0, 0:2], sharex=ax_scatter), fig.add_subplot(gs[1:3, 2], sharey=ax_scatter)
    sns.scatterplot(data=df, x=age_col, y='Fat_Total', hue='Gender', style='Gender', s=40, alpha=0.8, palette="Set2", ax=ax_scatter, edgecolors='black')
    for gender in df['Gender'].dropna().unique():
        sub = df[df['Gender'] == gender].dropna(subset=[age_col, 'Fat_Total'])
        if len(sub) > 1:
            m, c_val = np.polyfit(sub[age_col].values, sub['Fat_Total'].values, 1)
            ax_scatter.plot(sub[age_col].values, m * sub[age_col].values + c_val, ls='--', alpha=0.8, label=f'{gender} Fit')
    ax_scatter.set(xlabel='Age (years)', ylabel='Total Fatigue Score')
    ax_scatter.legend(fontsize=8, loc='best'); ax_scatter.grid(True, alpha=.3)
    ax_scatter.set_title("A. Joint Age vs. Fatigue", weight='bold')
    sns.kdeplot(data=df, x=age_col, hue='Gender', palette="Set2", ax=ax_x, fill=True, legend=False, alpha=0.4, common_norm=False)
    sns.kdeplot(data=df, y='Fat_Total', hue='Gender', palette="Set2", ax=ax_y, fill=True, legend=False, alpha=0.4, common_norm=False)
    ax_x.axis('off'); ax_y.axis('off')

    # B. 3D regression surfaces by gender
    ax_3d = fig.add_subplot(gs[0:3, 3:6], projection='3d')
    draw_3d_surface(df, age_col, ax_3d, title='B. 3D Demographic Surface by Gender')

    # C. Correlation heatmap
    ax_heat = fig.add_subplot(gs[3:6, 0:3])
    sns.heatmap(df[qcols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_heat, cbar=True, square=True, annot_kws={"size": 8}, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax_heat.set_title('C. Fatigue Item Correlation Heatmap', weight='bold', pad=30)
    ax_heat.set_xticklabels(qcols, rotation=0, ha='center', fontsize=8)
    ax_heat.set_yticklabels(qcols, rotation=0, fontsize=8)

    # D. Radar chart
    ax_radar = fig.add_subplot(gs[3:6, 3:6], polar=True)
    ax_radar.set_theta_offset(np.pi / 2); ax_radar.set_theta_direction(-1)
    angles = [n / 10.0 * 2 * np.pi for n in range(10)]; angles += angles[:1]
    plt.xticks(angles[:-1], qcols, size=8, color='black')
    ax_radar.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="black", size=8)
    plt.ylim(0, 5)
    for gender, color in [('Male', '#3498db'), ('Female', '#e74c3c')]:
        sub_radar = df[df['Gender'] == gender]
        if not sub_radar.empty:
            vals = sub_radar[qcols].mean().values.tolist()
            if vals and not any(np.isnan(vals)):
                vals += vals[:1]
                ax_radar.plot(angles, vals, linewidth=2, label=gender, color=color)
                ax_radar.fill(angles, vals, color=color, alpha=0.2)
    ax_radar.set_title('D. Fatigue Profile by Gender', weight='bold')
    ax_radar.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.35, 1.1))
    
    plt.savefig(FIG_ADVANCED, **SAVE_KW)
    return fig



def launch_gui_tabular(T, fig3):
    root = tk.Tk()
    root.title("GUI for Tables and 3D-vis")
    root.geometry("1100x850")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def add_tab(title, keys):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        for t, k in keys:
            make_table_view(scrollable_frame, t, T[k]).pack(fill=tk.X, pady=10, padx=10)

    frame_3d = ttk.Frame(notebook)
    notebook.add(frame_3d, text="Interactive 3D Visualisations")
    canvas_3d = FigureCanvasTkAgg(fig3, master=frame_3d)
    canvas_3d.draw()
    canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    root.mainloop()

def save_excel(df, tables, path):
    def _write(target):
        with pd.ExcelWriter(target, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            for name, tbl in tables.items():
                show_idx = tbl.index.name is not None or not isinstance(tbl.index, pd.RangeIndex)
                tbl.to_excel(writer, sheet_name=name[:31], index=show_idx)

    try:
        _write(path)
        print(f"\nDone -> {path} (Data + {len(tables)} summary table sheets)")
    except PermissionError:
        alt = path.replace('.xlsx', f'_{datetime.now():%Y%m%d_%H%M%S}.xlsx')
        try:
            _write(alt)
            print(f"\n[WARNING] {path} is locked. Saved instead -> {alt}")
        except Exception as e:
            print(f"\n[ERROR] Could not save output: {e}")
    except Exception as e:
        print(f"\n[WARNING] Could not save to {path}: {e}")

def main():
    result = process(INPUT)
    if result is None: return
    df, T, age_col, qcols = result

    fig_overview = plot_overview(df, age_col, qcols)
    fig_advanced = plot_advanced_analytics(df, age_col, qcols)
    fig_3d = plot_3d_standalone(df, age_col)
    fig_demo = plot_demographics_2x2(df, age_col)
    fig_fatigue = plot_fatigue_2x2(df, qcols)

    save_excel(df, T, OUTPUT)
    launch_gui_tabular(T, fig_3d)

#Please work
if __name__ == '__main__':
    main()