import pandas as pd
import numpy as np

DATA_PATH = r"C:\Users\praka\OneDrive\Desktop\CHINTA practice\Data compiler\merged_survey_data.xlsx"
DEMO_PATH = r"C:\Users\praka\OneDrive\Desktop\CHINTA practice\Demographics\output.xlsx"
FINAL_PATH = r"C:\Users\praka\OneDrive\Desktop\CHINTA practice\final_processed_ema.xlsx"

df = pd.read_excel(DATA_PATH, sheet_name='Valid Merged')
df['date'] = pd.to_datetime(df['date'])

# Day of week
dow_map = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
df['dow_name'] = df['date'].dt.dayofweek.map(dow_map)
print("Day of week counts:")
print(df['dow_name'].value_counts())

print(f"\nUnique participants: {df['Participant_No'].nunique()}")
print(f"Participant list: {sorted(df['Participant_No'].unique())}")

# Sleep columns
sleep_cols = [c for c in df.columns if 'sleep' in c.lower()]
print(f"\nSleep columns: {sleep_cols}")
for c in sleep_cols:
    print(f"  {c}: {df[c].notna().sum()} non-null")

# Demographics
df_demo = pd.read_excel(DEMO_PATH)
print(f"\nDemo shape: {df_demo.shape}")
print(f"Demo cols: {list(df_demo.columns)[:10]}")
print(f"P_No vals: {list(df_demo['P_No'].values)}")

# Check which surveys are missing per combo
has_s1 = df['S1_Workload_Mean'].notna()
has_s2 = df['S2_Fatigue_Mean'].notna()
has_s3 = df['S3_Leisure_Mean'].notna()

# For 2-survey rows, which survey is missing?
two_surveys = (has_s1.astype(int) + has_s2.astype(int) + has_s3.astype(int)) == 2
print(f"\n=== 2-SURVEY ROWS DETAIL ({two_surveys.sum()} rows) ===")
df_2 = df[two_surveys]
print(f"  Missing S1 (has S2+S3): {(~df_2['S1_Workload_Mean'].notna() & df_2['S2_Fatigue_Mean'].notna() & df_2['S3_Leisure_Mean'].notna()).sum()}")
print(f"  Missing S2 (has S1+S3): {(df_2['S1_Workload_Mean'].notna() & ~df_2['S2_Fatigue_Mean'].notna() & df_2['S3_Leisure_Mean'].notna()).sum()}")
print(f"  Missing S3 (has S1+S2): {(df_2['S1_Workload_Mean'].notna() & df_2['S2_Fatigue_Mean'].notna() & ~df_2['S3_Leisure_Mean'].notna()).sum()}")

# Sleep data availability
print(f"\n=== SLEEP DATA ===")
sleep_col_name = "S1_Ad'n: How would you rate your sleep quality last night? "
if sleep_col_name in df.columns:
    print(f"Sleep col found. Non-null: {df[sleep_col_name].notna().sum()}")
    print(f"Values: {df[sleep_col_name].dropna().unique()}")

# Weeks in data
df['week_num'] = df['date'].dt.isocalendar().week.astype(int)
print(f"\n=== WEEKS IN DATA ===")
print(f"Unique weeks: {sorted(df['week_num'].unique())}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Per-participant summary
print(f"\n=== PER-PARTICIPANT SURVEY COUNTS ===")
for p in sorted(df['Participant_No'].unique()):
    sub = df[df['Participant_No'] == p]
    n_total = len(sub)
    n3 = (sub['S1_Workload_Mean'].notna() & sub['S2_Fatigue_Mean'].notna() & sub['S3_Leisure_Mean'].notna()).sum()
    n2 = ((sub['S1_Workload_Mean'].notna().astype(int) + sub['S2_Fatigue_Mean'].notna().astype(int) + sub['S3_Leisure_Mean'].notna().astype(int)) == 2).sum()
    n1 = ((sub['S1_Workload_Mean'].notna().astype(int) + sub['S2_Fatigue_Mean'].notna().astype(int) + sub['S3_Leisure_Mean'].notna().astype(int)) == 1).sum()
    print(f"  {p}: {n_total} days total | 3-survey={n3}, 2-survey={n2}, 1-survey={n1}")
