import pandas as pd
import numpy as np
import warnings
import os
import subprocess

warnings.filterwarnings("ignore")

#PATH DEFINITION
BASE_DIR = r"C:\Users\praka\OneDrive\Desktop\CHINTA practice"
DATA_PATH = os.path.join(BASE_DIR, "Data compiler", "merged_survey_data.xlsx")
DEMO_PATH = os.path.join(BASE_DIR, "Demographics", "output.xlsx")

FORMATTED_DIR = os.path.join(BASE_DIR, "Formatted_Data")
os.makedirs(FORMATTED_DIR, exist_ok=True)

ORIGINAL_OUTPUT = os.path.join(FORMATTED_DIR, "original_with_missing.xlsx")
COMPLETE_OUTPUT = os.path.join(FORMATTED_DIR, "original_complete_cases.xlsx")
TEMP_IMPUTE = os.path.join(FORMATTED_DIR, "temp_for_imputation.parquet")
IMPUTED_RAW = os.path.join(FORMATTED_DIR, "imputed_m5_datasets.parquet")
IMPUTED_PROCESSED = os.path.join(FORMATTED_DIR, "imputed_processed_ema.parquet")

class DataQualityReport:
    def __init__(self):
        self.lines = []
        
    def add(self, text=""):
        self.lines.append(text)
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode('ascii', errors='replace').decode('ascii'))
        
    def add_section(self, title):
        self.add("\n")
        self.add(f"  {title}")
        self.add("\n")
        
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))


class EMADataset:
    #Handles the robust loading, merging, imputation and standardisation of EMA data.
    def __init__(self):
        self.report = DataQualityReport()
    
    def load_and_prepare(self):
        self.report.add_section("STAGE 1: ORGANISATION & IMPUTATION")
        
        # STEP 1: Load raw data
        self.report.add("\n[Step 1/8] Loading raw datasets")
        df_raw = pd.read_excel(DATA_PATH, sheet_name="Valid Merged")
        df_demo = pd.read_excel(DEMO_PATH)
        self.report.add(f"  Raw merged survey rows: {len(df_raw)}")
        
        
        # STEP 2: Fix participant count
        verified_participants = set(df_demo['P_No'].unique())
        df = df_raw[df_raw['Participant_No'].isin(verified_participants)].copy()
        self.report.add(f"  After filtering to verified: {len(df)} rows, {df['Participant_No'].nunique()} participants")
        
        # Merge demographics (Trait Fatigue)
        df_demo_subset = df_demo[['P_No', 'Fat_Total']].rename(
            columns={'P_No': 'Participant_No', 'Fat_Total': 'Trait_Fatigue'})
        df = df.merge(df_demo_subset, on='Participant_No', how='left')
        
        # STEP 3: Temporal features
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['Participant_No', 'date']).reset_index(drop=True)
        
        # Filter to only keep week 1 and 2 for analyses
        min_date = df['date'].min()
        df['study_week'] = df['date'].apply(lambda d: ((d - min_date).days // 7) + 1)
        df = df[df['study_week'] <= 2].copy()
        self.report.add(f"  Excluded Week 3 data. Retained: {len(df)} rows (Week 1 and 2 only)")
        
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Mon 6=Sun
        
        df['is_week_onset'] = (df['day_of_week'] == 0).astype(int)
        df['is_week_finish'] = (df['day_of_week'] == 4).astype(int)
        df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
        df['week_id'] = df['date'].dt.isocalendar().week.astype(int)
        
        # Extract Sleep Quality
        sleep_col = "S1_Ad'n: How would you rate your sleep quality last night? "
        if sleep_col in df.columns:
            df['Sleep_Quality'] = pd.to_numeric(df[sleep_col], errors='coerce')
        else:
            df['Sleep_Quality'] = np.nan
            
        # Transform-then-Impute for Leisure (create early for bounds/descriptives)/as suggested
        '''Reason: leisure is skewed, the imputer assumes normality, so log-transforming before imputing produces 
        better statistical draws and guarantees the back-transformed values stay non-negative — rather than 
        imputing on a skewed raw scale and hoping clamping fixes the fallout afterward.'''
        if 'S3_Leisure_Mean' in df.columns:
            df['Leisure_Log'] = np.log1p(df['S3_Leisure_Mean'])
            
        # MISSINGNESS DIAGNOSTICS (MAR AUDIT)
        self.report.add_section("MISSING DATA (MAR)")
        for var in ['S1_Workload_Mean', 'Leisure_Log', 'Sleep_Quality', 'S2_Fatigue_Mean']:
            missing_pct = df[var].isna().mean() * 100
            self.report.add(f"Overall missingness in {var}: {missing_pct:.2f}%")
            
            # By Day of Week
            dow_missing = df.groupby('day_of_week')[var].apply(lambda x: x.isna().mean() * 100)
            self.report.add("  Missingness by day of week (0=Mon, 6=Sun):")
            for dow, val in dow_missing.items():
                self.report.add(f"    Day {dow}: {val:.2f}%")
                
            # By Week Onset/Finish
            onset_missing = df.groupby('is_week_onset')[var].apply(lambda x: x.isna().mean() * 100).to_dict()
            finish_missing = df.groupby('is_week_finish')[var].apply(lambda x: x.isna().mean() * 100).to_dict()
            self.report.add(f"  Missingness by Week Onset (Monday): {onset_missing}")
            self.report.add(f"  Missingness by Week Finish (Friday): {finish_missing}")
            
            # By Workload level (where observed)
            if var != 'S1_Workload_Mean':
                wl_med = df['S1_Workload_Mean'].median()
                df_temp = df.dropna(subset=['S1_Workload_Mean']).copy()
                df_temp['high_workload'] = (df_temp['S1_Workload_Mean'] > wl_med).astype(int)
                wl_missing = df_temp.groupby('high_workload')[var].apply(lambda x: x.isna().mean() * 100)
                self.report.add(f"  Missingness in {var} by Workload level (where observed) (0=Low, 1=High):")
                for wl, val in wl_missing.items():
                    self.report.add(f"    Workload {wl}: {val:.2f}%")

        # Survey completion filters (drop 0-1 survey days)
        df['has_S1'] = df['S1_Workload_Mean'].notna().astype(int)
        df['has_S2'] = df['S2_Fatigue_Mean'].notna().astype(int)
        df['has_S3'] = df['S3_Leisure_Mean'].notna().astype(int)
        df['surveys_filled'] = df['has_S1'] + df['has_S2'] + df['has_S3']
        
        df = df[df['surveys_filled'] >= 2].copy()
        self.report.add(f"\n  Retained 2+ survey days: {len(df)} rows")
        
        # Identifiers P_NO
        df['Part_Day'] = df['Participant_No'].astype(str) + "_" + df['day_number'].astype(str)
        df['Part_Week'] = df['Participant_No'].astype(str) + "_W" + df['week_id'].astype(str)
                
        # STEP 4: Capture clamping bounds from raw observed data
        bounds = {
            'Sleep_Quality': (df['Sleep_Quality'].min(), df['Sleep_Quality'].max()),
            'S1_Workload_Mean': (df['S1_Workload_Mean'].min(), df['S1_Workload_Mean'].max()),
            'S1_Workload_SD': (df['S1_Workload_SD'].min(), df['S1_Workload_SD'].max())
        }
        if 'Leisure_Log' in df.columns:
            bounds['Leisure_Log'] = (df['Leisure_Log'].min(), df['Leisure_Log'].max())
        
        # ------------------------------------------------------------------
        # STEP 5: Export Original Data (With Missing and Complete Cases)
        # ------------------------------------------------------------------
        self.report.add("\n[Step 5/8] Saving original datasets to Formatted_Data...")
        df_original = self._apply_centering_and_transforms(df.copy(), recompute_interactions=True)
        
        # 1. Original with missing
        # Write stats to an extra sheet
        with pd.ExcelWriter(ORIGINAL_OUTPUT) as writer:
            df_original.to_excel(writer, sheet_name='Data', index=False)
            stats_df = pd.DataFrame({'Metric': ['Total Rows', 'Unique Participants'], 'Value': [len(df_original), df_original['Participant_No'].nunique()]})
            stats_df.to_excel(writer, sheet_name='Stats', index=False)
        self.report.add(f"  Saved {ORIGINAL_OUTPUT}")
        
        # 2. Original complete cases (drop any NaNs in core variables)
        core_vars = ['S2_Fatigue_Mean_z', 'Workload_WP_z', 'Leisure_WP_z', 'Sleep_WP_z']
        df_complete = df_original.dropna(subset=[c for c in core_vars if c in df_original.columns])
        with pd.ExcelWriter(COMPLETE_OUTPUT) as writer:
            df_complete.to_excel(writer, sheet_name='Data', index=False)
            stats_df = pd.DataFrame({'Metric': ['Complete Cases Rows', 'Unique Participants'], 'Value': [len(df_complete), df_complete['Participant_No'].nunique()]})
            stats_df.to_excel(writer, sheet_name='Stats', index=False)
        self.report.add(f"  Saved {COMPLETE_OUTPUT}")

        # STEP 5b: 3-Level ICC Variance Check on Complete Cases
        self.report.add_section("3-LEVEL ICC VARIANCE CHECK (Complete Cases)")
        try:
            if 'S2_Fatigue_Mean_z' in df_complete.columns and 'Part_Week' in df_complete.columns:
                # Compute grand mean
                grand_mean = df_complete['S2_Fatigue_Mean_z'].mean()
                
                # Between-person variance (Level 3): variance of person means
                person_means = df_complete.groupby('Participant_No')['S2_Fatigue_Mean_z'].mean()
                var_L3_person = float(person_means.var(ddof=1)) if len(person_means) > 1 else 0.0
                
                # Between-week-within-person variance (Level 2): variance of
                # week means, after removing the person-level component
                week_means = df_complete.groupby('Part_Week')['S2_Fatigue_Mean_z'].mean()
                person_of_week = df_complete.groupby('Part_Week')['Participant_No'].first()
                week_person_means = person_of_week.map(person_means)
                week_deviations = week_means - week_person_means
                var_L2_week = float(week_deviations.var(ddof=1)) if len(week_deviations) > 1 else 0.0
                
                # Within-week variance (Level 1): residual variance within weeks
                week_person_mapped = df_complete['Part_Week'].map(week_means)
                residuals_L1 = df_complete['S2_Fatigue_Mean_z'] - week_person_mapped
                var_L1_within = float(residuals_L1.var(ddof=1))
                
                total_var = var_L3_person + var_L2_week + var_L1_within
                if total_var > 0:
                    icc_person = var_L3_person / total_var
                    icc_week   = var_L2_week   / total_var
                    icc_within = var_L1_within / total_var
                else:
                    icc_person = icc_week = icc_within = 0.0
                
                self.report.add(f"  Between-Person (L3)   variance share: {icc_person*100:.2f}%  (ICC_person = {icc_person:.4f})")
                self.report.add(f"  Between-Week-in-Person (L2) var share: {icc_week*100:.2f}%  (ICC_week = {icc_week:.4f})")
                self.report.add(f"  Within-Week (L1)       variance share: {icc_within*100:.2f}%")
                self.report.add(f"  Total n_complete = {len(df_complete)}, n_participants = {df_complete['Participant_No'].nunique()}, n_weeks = {df_complete['Part_Week'].nunique()}")
                
                if icc_week < 0.05 or (var_L2_week / total_var) < 0.01:
                    self.report.add("  DECISION: Week-level variance < 5% (ICC_3 < 0.05 threshold).")
                    self.report.add("            3-Level models are DROPPED. Use 2-Level models only.")
                    self.report.add("            This ensures congeniality with the 2-level MICE imputer.")
                else:
                    self.report.add("  DECISION: Week-level variance >= 5%. 3-Level models are RETAINED.")
                    
        except Exception as e:
            self.report.add(f"  [WARNING] 3L ICC variance check failed: {e}")
        
        # STEP 6: Execute R-Py Bridge for Imputation
        self.report.add("\n[Step 6/8] Executing R-Py Bridge for FCS-LMM Imputation...")
        # For JAV: pre-calculate interactions on df_original (which has NaNs)
        df_for_r = df_original.copy()
        if 'S3_Leisure_Mean' in df_for_r.columns:
            df_for_r = df_for_r.drop(columns=['S3_Leisure_Mean'])
            
        df_for_r.to_parquet(TEMP_IMPUTE, index=False)
        
        try:
            rscript_path = r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
            subprocess.run([rscript_path, "impute_engine.R"], check=True)
            self.report.add("  R imputation script completed successfully.")
        except subprocess.CalledProcessError as e:
            self.report.add(f"  ERROR: R script failed. Make sure R is installed and in PATH. {e}")
            return None
        except FileNotFoundError:
            self.report.add("  ERROR: Rscript not found. Make sure R is installed and in PATH.")
            return None
            
        # Load m=20 imputed datasets
        df_imputed_long = pd.read_parquet(IMPUTED_RAW)
        self.report.add(f"  Loaded imputed datasets: {len(df_imputed_long)} rows (m=20)")
        
        # Restore NA for outcome variable (MIDO strategy)
        df_imputed_long['S2_Fatigue_Mean'] = df_imputed_long['Part_Day'].map(df.set_index('Part_Day')['S2_Fatigue_Mean'])
        self.report.add("  Restored original NA values for S2_Fatigue_Mean (MIDO strategy).")
        
        # STEP 7: Clamping Bounds
        self.report.add("\n[Step 7/8] Clamping imputed values to mathematical bounds...")
        for col, (b_min, b_max) in bounds.items():
            if col in df_imputed_long.columns:
                df_imputed_long[col] = df_imputed_long[col].clip(lower=b_min, upper=b_max)
                self.report.add(f"  Clamped {col} to [{b_min}, {b_max}]")
                
        # Prev_Leisure Day-1 fill is no longer on S3_Leisure_Mean but on Leisure_Log
        df_imputed_long['Prev_Leisure'] = df_imputed_long.groupby(['.imp', 'Participant_No'])['Leisure_Log'].shift(1)
        part_leisure_means = df_imputed_long.groupby(['.imp', 'Participant_No'])['Leisure_Log'].transform('mean')
        day1_mask = df_imputed_long['Prev_Leisure'].isna() & df_imputed_long['Leisure_Log'].notna()
        df_imputed_long.loc[day1_mask, 'Prev_Leisure'] = part_leisure_means[day1_mask]
        
        # STEP 8: Process all m=20 datasets (Centering & Transforms)
        self.report.add("\n[Step 8/8] Processing centering and transforms for all m=20 datasets...")
        
        processed_datasets_jav = []
        processed_datasets_passive = []
        for m in sorted(df_imputed_long['.imp'].unique()):
            df_m = df_imputed_long[df_imputed_long['.imp'] == m].copy()
            df_m['S3_Leisure_Mean'] = np.expm1(df_m['Leisure_Log'])
            
            # JAV path: Use the imputed interaction columns directly (recompute_interactions=False)
            df_m_jav = self._apply_centering_and_transforms(df_m.copy(), recompute_interactions=False)
            processed_datasets_jav.append(df_m_jav)
            
            # Passive path: Recompute interaction columns within each dataset
            df_m_passive = self._apply_centering_and_transforms(df_m.copy(), recompute_interactions=True)
            processed_datasets_passive.append(df_m_passive)
            
        df_final_pooled_jav = pd.concat(processed_datasets_jav, ignore_index=True)
        df_final_pooled_passive = pd.concat(processed_datasets_passive, ignore_index=True)
        
        IMPUTED_PROCESSED_JAV = os.path.join(FORMATTED_DIR, "imputed_processed_ema_jav.parquet")
        IMPUTED_PROCESSED_PASSIVE = os.path.join(FORMATTED_DIR, "imputed_processed_ema_passive.parquet")
        
        df_final_pooled_jav.to_parquet(IMPUTED_PROCESSED_JAV, index=False)
        df_final_pooled_passive.to_parquet(IMPUTED_PROCESSED_PASSIVE, index=False)
        
        # Write default file too for backward compatibility
        df_final_pooled_passive.to_parquet(IMPUTED_PROCESSED, index=False)
        
        self.report.add(f"  Saved JAV processed datasets to {IMPUTED_PROCESSED_JAV}")
        self.report.add(f"  Saved Passive processed datasets to {IMPUTED_PROCESSED_PASSIVE}")
        
        # JAV Internal Consistency Diagnostic
        orig_missing_mask = df_original['Interaction_Normal'].isna()
        missing_days = df_original.loc[orig_missing_mask, 'Part_Day'].unique()
        df_imputed_missing_rows = df_final_pooled_jav[df_final_pooled_jav['Part_Day'].isin(missing_days)]
        
        if len(df_imputed_missing_rows) > 0:
            recomputed_normal = df_imputed_missing_rows['Workload_Baseline_WP'] * df_imputed_missing_rows['Leisure_Log_WP_z']
            recomputed_crunch = df_imputed_missing_rows['Workload_Crunch_WP'] * df_imputed_missing_rows['Leisure_Log_WP_z']
            
            mae_normal = np.mean(np.abs(df_imputed_missing_rows['Interaction_Normal'] - recomputed_normal))
            mae_crunch = np.mean(np.abs(df_imputed_missing_rows['Interaction_Crunch'] - recomputed_crunch))
            
            self.report.add_section("JAV INTERNAL CONSISTENCY")
            self.report.add(f"Mean Absolute Discrepancy for Imputed Interaction_Normal: {mae_normal:.4f}")
            self.report.add(f"Mean Absolute Discrepancy for Imputed Interaction_Crunch: {mae_crunch:.4f}")
            
            disc_count_normal = np.sum(np.abs(df_imputed_missing_rows['Interaction_Normal'] - recomputed_normal) > 0.2)
            self.report.add(f"Number of rows with Interaction_Normal discrepancy > 0.2: {disc_count_normal} / {len(df_imputed_missing_rows)}")

        # --- PASSIVE INTERNAL CONSISTENCY DIAGNOSTIC ---
        # In the passive dataset, interactions are always recomputed as products
        # of their components, so discrepancy should be exactly 0.
        # Any deviation indicates a bug in the centering/transform pipeline.
        if os.path.exists(IMPUTED_PROCESSED_PASSIVE):
            df_passive_check = pd.read_parquet(IMPUTED_PROCESSED_PASSIVE)
            if all(c in df_passive_check.columns for c in
                   ['Interaction_Normal', 'Interaction_Crunch',
                    'Workload_Baseline_WP', 'Workload_Crunch_WP', 'Leisure_Log_WP_z']):
                recomp_n = df_passive_check['Workload_Baseline_WP'] * df_passive_check['Leisure_Log_WP_z']
                recomp_c = df_passive_check['Workload_Crunch_WP']  * df_passive_check['Leisure_Log_WP_z']
                mae_p_n = np.mean(np.abs(df_passive_check['Interaction_Normal'] - recomp_n))
                mae_p_c = np.mean(np.abs(df_passive_check['Interaction_Crunch']  - recomp_c))
                disc_p  = np.sum(np.abs(df_passive_check['Interaction_Normal'] - recomp_n) > 1e-9)

                self.report.add_section("PASSIVE INTERNAL CONSISTENCY")
                self.report.add(f"Mean Absolute Discrepancy for Interaction_Normal: {mae_p_n:.6f}  (expected: 0.000000)")
                self.report.add(f"Mean Absolute Discrepancy for Interaction_Crunch:  {mae_p_c:.6f}  (expected: 0.000000)")
                if disc_p == 0:
                    self.report.add("PASS: All interactions match their component products exactly.")
                else:
                    self.report.add(f"WARNING: {disc_p} rows have non-zero discrepancy — check centering pipeline.")
        
        # Cleanup
        if os.path.exists(TEMP_IMPUTE):
            os.remove(TEMP_IMPUTE)
            
        report_path = os.path.join(FORMATTED_DIR, "data_quality_report.txt")
        self.report.save(report_path)
        
        return df_final_pooled_passive

    def _apply_centering_and_transforms(self, df, recompute_interactions=True):
        """Applies person-mean centering, standardisation, and non-linear transforms."""
        
        # Person-Mean Centering
        centering_vars = {
            'Workload': 'S1_Workload_Mean',
            'Leisure': 'S3_Leisure_Mean',
            'Leisure_Log': 'Leisure_Log',
            'Sleep': 'Sleep_Quality',
        }
        
        for prefix, raw_col in centering_vars.items():
            if raw_col not in df.columns: continue
            
            df[f'{prefix}_BP'] = df.groupby('Participant_No')[raw_col].transform('mean')
            df[f'{prefix}_WP'] = df[raw_col] - df[f'{prefix}_BP']
            
            for suffix in ['WP', 'BP']:
                col = f'{prefix}_{suffix}'
                col_std = df[col].std()
                df[f'{col}_z'] = (df[col] - df[col].mean()) / col_std if col_std > 0 else 0.0
        
        # Standardise others
        if 'S1_Workload_SD' in df.columns:
            sd_std = df['S1_Workload_SD'].std()
            df['S1_Workload_SD_z'] = (df['S1_Workload_SD'] - df['S1_Workload_SD'].mean()) / sd_std if sd_std > 0 else 0.0
            
        fatigue_std = df['S2_Fatigue_Mean'].std()
        df['S2_Fatigue_Mean_z'] = (df['S2_Fatigue_Mean'] - df['S2_Fatigue_Mean'].mean()) / fatigue_std if fatigue_std > 0 else 0.0
        
        trait_std = df['Trait_Fatigue'].std()
        df['Trait_Fatigue_z'] = (df['Trait_Fatigue'] - df['Trait_Fatigue'].mean()) / trait_std if trait_std > 0 else 0.0
        
        dn_std = df['day_number'].std()
        df['day_number_z'] = (df['day_number'] - df['day_number'].mean()) / dn_std if dn_std > 0 else 0.0
        
        # Non-Linear Transforms
        if 'Workload_WP_z' in df.columns:
            df['Workload_Baseline_WP'] = np.minimum(0, df['Workload_WP_z'])
            df['Workload_Crunch_WP'] = np.maximum(0, df['Workload_WP_z'])
        
        if 'S3_Leisure_Mean' in df.columns:
            df['Leisure_Log'] = np.log1p(df['S3_Leisure_Mean'])
            df['Leisure_Log_BP'] = df.groupby('Participant_No')['Leisure_Log'].transform('mean')
            df['Leisure_Log_WP'] = df['Leisure_Log'] - df['Leisure_Log_BP']
            
            for suffix in ['WP', 'BP']:
                col = f'Leisure_Log_{suffix}'
                col_std = df[col].std()
                df[f'{col}_z'] = (df[col] - df[col].mean()) / col_std if col_std > 0 else 0.0
                
        if recompute_interactions:
            if 'Workload_Baseline_WP' in df.columns and 'Leisure_Log_WP_z' in df.columns:
                df['Interaction_Normal'] = df['Workload_Baseline_WP'] * df['Leisure_Log_WP_z']
                df['Interaction_Crunch'] = df['Workload_Crunch_WP'] * df['Leisure_Log_WP_z']
            
        return df

def main():
    print("=" * 70)
    print("  PIPELINE STAGE 1: DATA ORGANISATION & FCS-LMM IMPUTATION")
    print("=" * 70)
    
    dataset = EMADataset()
    df = dataset.load_and_prepare()
    
    print("\n" + "=" * 70)
    print(f"  STAGE 1 COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
