import pandas as pd
import hashlib
import matplotlib.pyplot as plt

def process_clinical_data(input_file, output_file ):
    print("Loading data...")
    df = pd.read_excel(r"C:\Users\praka\OneDrive\Desktop\CHINTA practice\EMA consent form and Trait-behaviour (Responses).xlsx")
    
# Step 1. Confirming consent of the participant: Filter only those who answered starting with "Yes"
    initial_count = len(df)
    col_b = df.columns[1]  # "Have you read the ICF column"
    col_c = df.columns[2]  # "Do you consent to participate column"
    
    has_read_icf = df[col_b].astype(str).str.strip().str.lower().str.startswith('yes')
    has_consented = df[col_c].astype(str).str.strip().str.lower().str.startswith('yes')
    
    df = df[has_read_icf & has_consented]
    
    print(f"Filtering out {initial_count - len(df)} participants who did not consent in Columns B and C.")
    print(f"Total number of consenting participants are:{len(df)}")

    if len(df) == 0:
        print("No consenting participants found. Stopping execution.")
        return


# Step 2. Anonymize Participant Names
def anaonymise(df): 
    name_col = df.columns[3] 
    df['Participant_Number'] = [f"Participant_{i:03d}" for i in range(1, len(df) + 1)]

    def generate_numeric_id(name):
        if pd.isna(name): return None

        name_bytes = str(name).strip().lower().encode('utf-8')
        # Create a SHA-256 hash and convert to 8-digit integer
        hash_hex = hashlib.sha256(name_bytes).hexdigest()
        return int(hash_hex, 16) % 10**8
        
    df['Participant_ID'] = df[name_col].apply(generate_numeric_id)
    
    #Reordering the colums for them to sit adjacent 
    all_cols = list(df.columns)
    new_col_order = ['Participant_Number', 'Participant_ID'] + [c for c in all_cols if c not in ['Participant_Number', 'Participant_ID']]
    df = df[new_col_order]
    

# #Step 3. Age Stats
    age_col = df.columns[5] 
    df[age_col] = pd.to_numeric(df[age_col], errors='coerce')  
    # Calculate quartiles
    q1 = df[age_col].quantile(0.25)
    q3 = df[age_col].quantile(0.75)
    iqr = q3 - q1
    
    # Age Grouping (This based on my intuition in this moment (lol))
    age_bins = [18, 20, 25, 30, 35, 40, 45, 50, 65, 100]
    age_labels = ['18-20', '20-25', '25-30', '30-35', '40-45', '45-50', '>50']
    df['Age_Group'] = pd.cut(df[age_col], bins=age_bins, labels=age_labels, right=True)
    age_group_counts = df['Age_Group'].value_counts().sort_index().to_dict()
    


    age_stats = {
        "Mean Age": df[age_col].mean(),
        "Median Age": df[age_col].median(),
        "Max Age": df[age_col].max(),
        "Min Age": df[age_col].min(),
        "Standard Deviation": df[age_col].std(),
        "Variance": df[age_col].var(),
        "25th Percentile (Q1)": q1,
        "75th Percentile (Q3)": q3,
        "Interquartile Range (IQR)": iqr,
        "Skewness": df[age_col].skew()
    }
    
    # 5. Gender Demographics
    gender_col = df.columns[6] 
    
    def parse_gender(response):
        if pd.isna(response): return 'Other/Unknown'
        first_letter = str(response).strip().lower()[0]
        if first_letter == 'm': return 'Male'
        if first_letter == 'f': return 'Female'
        return 'Other/Unknown'
          
    df['Parsed_Gender'] = df[gender_col].apply(parse_gender)
    
    # 5.1 Distribution & Percentages
    gender_counts = df['Parsed_Gender'].value_counts()
    gender_percentages = df['Parsed_Gender'].value_counts(normalize=True) * 100
    
    gender_stats_detailed = {}
    for gender in gender_counts.index:
        gender_stats_detailed[gender] = {
            "Count": int(gender_counts[gender]),
            "Percentage": f"{gender_percentages[gender]:.1f}%"
        }

    # 5.2 Cross-Tabulation: Gender vs Age Group Intersection
    if 'Age_Group' in df.columns:
        gender_age_crosstab = pd.crosstab(df['Parsed_Gender'], df['Age_Group']).to_dict(orient='index')
    else:
        gender_age_crosstab = "Age data missing or not computed yet."

    # 5.3 Trait Stress/Fatigue Mapping by Gender
    if 'Total_Fatigue_Score' in df.columns:
        fatigue_by_gender = df.groupby('Parsed_Gender')['Total_Fatigue_Score'].mean().to_dict()
    else:
        fatigue_by_gender = "Total fatigue scores missing or not graded yet."


    # 6. Education Statistics
    edu_col = df.columns[7]
    
    # Standardize responses into sequential rankings (Ordinal Mapping)
    education_rank_mapping = {
        "no formal education": 1,
        "high school diploma or equivalent (10th/12th)": 2,
        "associate degree or vocational training": 3,
        "bachelor’s degree": 4,
        "master’s degree": 5,
        "doctorate or professional degree (e.g., phd, md, jd)": 6,
        "prefer not to answer": None
    }
    # Strip whitespace and convert to lowercase 
    df['Education_Cleaned'] = df[edu_col].astype(str).str.strip().str.lower()
    df['Education_Rank'] = df['Education_Cleaned'].map(education_rank_mapping)
    
    #6.1 Frequency Distribution & Sample Percentages
    edu_counts = df['Education_Cleaned'].value_counts()
    edu_percentages = df['Education_Cleaned'].value_counts(normalize=True) * 100
    
    edu_demographics = {}
    for tier in edu_counts.index:
        edu_demographics[tier] = {
            "Count": int(edu_counts[tier]),
            "Percentage": f"{edu_percentages[tier]:.1f}%"
        }
        
    #6.2 Basic Cohort Central Tendency  (e.g., What is the average education level of our study group?)
    avg_edu_rank = df['Education_Rank'].mean() # this is a score between 1 and 6
    median_edu_rank = df['Education_Rank'].median()
    
    # 6.3. Psychometric Interaction: Mean Cognitive Fatigue by Education Level
    if 'Total_Fatigue_Score' in df.columns:
        fatigue_by_education = df.groupby('Education_Cleaned')['Total_Fatigue_Score'].mean().to_dict()
    else:
        fatigue_by_education = "Fatigue scores not found."

    #Print
    print(f"Cohort Median Education Rank: {median_edu_rank:.1f} / 6.0")
    print(f"\n")
    print("\nEducation Tier Distribution:")
    for tier, data in edu_demographics.items():
        print(f"* {tier.title()}: {data['Count']} ({data['Percentage']})")
        
    print("\n**Mean Fatigue Score by Education Group:**")
    if isinstance(fatigue_by_education, dict):
        for tier, mean_score in fatigue_by_education.items():
            print(f"* {tier.title()}: {mean_score:.2f}")

 
 # 7.Likert-Scale vectorised scoring
    start_idx = 8
    num_questions = 10
    likert_cols = list(df.columns[start_idx : start_idx + num_questions])
    
    #Likert Mapping dictionary
    likert_mapping = {
        "doesn't apply at all": 1,
        "does not apply much": 2,
        "slightly applies": 3,
        "applies a lot": 4,
        "applies completely": 5
    }
    
    df_cleaned_text = df[likert_cols].astype(str).apply(lambda x: x.str.strip().str.lower())
    df_scores = df_cleaned_text.map(likert_mapping)
    
    # Rename 
    score_col_names = [f"Q{i}_Score" for i in range(1, num_questions + 1)]
    df[score_col_names] = df_scores

    # Finding indivisual metrics 
    
    # 7.1 Total & Mean Trait Fatigue Scores
    df['Total_Fatigue_Score'] = df[score_col_names].sum(axis=1)
    df['Mean_Fatigue_Score'] = df[score_col_names].mean(axis=1)
    
    # 7.2 Within-Subject Variability (Standard Deviation per person)
    df['Fatigue_Response_SD'] = df[score_col_names].std(axis=1)
    
    # 7.3 Extreme Response Counting (Count of 5s per person)
    df['Extreme_Fatigue_Indicators'] = (df[score_col_names] == 5).sum(axis=1)

    # 7.4 Calculate average fatigue across the entire study population
    cohort_mean_fatigue = df['Total_Fatigue_Score'].mean()
    cohort_sd_fatigue = df['Total_Fatigue_Score'].std()
    
    print("\n")
    print(f"Cohort Baseline Fatigue Level: {cohort_mean_fatigue:.2f} (± {cohort_sd_fatigue:.2f})")
    
    # Save the highly detailed master file
    df.to_excel(f"C:\Users\praka\OneDrive\Desktop\CHINTA practice\output.xlsx", index=False)
    print(f"Data pipeline complete. Advanced profile saved to: file")


#Execution 
if __name__ == "__main__":
    import os
    
    input_file = r"C:\Users\praka\OneDrive\Desktop\CHINTA practice\EMA consent form and Trait-behaviour (Responses).xlsx"
    output_file = f"C:\Users\praka\OneDrive\Desktop\CHINTA practice\output.xlsx"
    
    
    try:
        
        results = process_clinical_study(input_file, output_file)
        
        if results is None:
            print("Processing stopped: No valid consenting data found.")
            exit()
            
        df_final, age_stats, age_group_counts, gender_stats_detailed, edu_demographics, psych_stats = results
        

        print("\n" + "="*50)
        print("          CLINICAL STUDY POPULATION REPORT          ")
        print("="*50)
        
        print("\n[1] ADVANCED AGE METRICS")
        print(f"{"Metric":<30} | {"Value":<15}")
        print("-" * 48)
        for stat, val in age_stats.items():
            val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
            print(f"{stat:<30} | {val_str:<15}")
            
        print("\n[2] GENDER DEMOGRAPHICS & OUTCOME DISTRIBUTION")
        print(f"{"Gender Group":<15} | {"Cohort Count":<12} | {"Cohort %":<10}")
        print("-" * 45)
        for gender, data in gender_stats_detailed.items():
            print(f"{gender:<15} | {data['Count']:<12} | {data['Percentage']:<10}")
            
        print("\n[3] EDUCATION & COGNITIVE RESERVE TIERS")
        print(f"{"Education Level Completed":<45} | {"Count":<8} | {"Percentage":<10}")
        print("-" * 68)
        for tier, data in edu_demographics.items():
            print(f"{tier.title():<45} | {data['Count']:<8} | {data['Percentage']:<10}")
            
        print("\n[4] BASELINE TRAIT PSYCHOMETRICS")
        for k, v in psych_stats.items():
            print(f"{k:<30} : {v:.2f}")
        print("="*50)


        print("\nGenerating Consolidated Visual Dashboard:")
        
        # Setup a 2x2 grid layout for an all-in-one metrics dashboard
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Clinical Population Profile & Psychometric Distribution', fontsize=18, weight='bold', y=0.96)
        
        # Panel A: Age Distribution Histogram + Kernel Density Approximation
        axes[0, 0].hist(df_final[df_final.columns[5]].dropna(), bins=8, color='#4C72B0', edgecolor='black', alpha=0.8)
        axes[0, 0].set_title('A: Cohort Age Demographics Spread', fontsize=12, weight='bold')
        axes[0, 0].set_xlabel('Age Units (Years)')
        axes[0, 0].set_ylabel('Participant Count')
        axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)
        
        # Panel B: Gender Proportions Pie Chart
        gender_labels = list(gender_stats_detailed.keys())
        gender_sizes = [data['Count'] for data in gender_stats_detailed.values()]
        axes[0, 1].pie(gender_sizes, labels=gender_labels, autopct='%1.1f%%', startangle=90, 
                       colors=['#DD8452', '#55A868', '#C44E52'], wedgeprops={'edgecolor': 'black', 'linewidth': 1})
        axes[0, 1].set_title('B: Gender Representation Balance', fontsize=12, weight='bold')
        
        # Panel C: Education Stratification Horizontal Bar Chart
        edu_labels = [t.title()[:30] + '...' if len(t) > 30 else t.title() for t in edu_demographics.keys()]
        edu_sizes = [data['Count'] for data in edu_demographics.values()]
        axes[1, 0].barh(edu_labels, edu_sizes, color='#8172B3', edgecolor='black')
        axes[1, 0].set_title('C: Educational Background Tiers', fontsize=12, weight='bold')
        axes[1, 0].set_xlabel('Count')
        axes[1, 0].invert_yaxis()  # Keeps top level at the top
        axes[1, 0].grid(axis='x', linestyle='--', alpha=0.5)
        
        # Panel D: Total Fatigue Scores Boxplot
        axes[1, 1].boxplot(df_final['Total_Fatigue_Score'].dropna(), vert=False, patch_artist=True,
                           boxprops=dict(facecolor='#CCB974', color='black'),
                           medianprops=dict(color='darkred', linewidth=2))
        axes[1, 1].set_title('D: Cohort Baseline Trait Fatigue Spread', fontsize=12, weight='bold')
        axes[1, 1].set_xlabel('Total Score Array (Likert Cumulative Metric Matrix)')
        axes[1, 1].set_yticklabels(['Study Cohort'])
        axes[1, 1].grid(axis='x', linestyle='--', alpha=0.5)
        
        # Fine-tune layouts and save to file
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        dashboard_output = 'clinical_demographics_dashboard.png'
        plt.savefig(dashboard_output, dpi=300)
        plt.close()
        
        print(f"-> Success! Composite metrics dashboard graphics exported to: '{dashboard_output}'")
        
    except FileNotFoundError:
        print(f"\n[Execution Error]: Target file '{input_file}' is missing from the directory.")
    except Exception as e:
        print(f"\n[Runtime Exception Encountered]: {str(e)}")