import subprocess
import sys
import time

stages = [
    ("1_organisation.py", "Data Formatting & Multilevel Imputation"),
    ("2_frequentist_linear.py", "Frequentist Linear Mixed Models (with Rubin's Rules)"),
    ("3_bayesian_linear.py", "Bayesian Linear Mixed Models (with posterior pooling)"),
    ("4_model_comparison.py", "Core Model Comparison Report (4 Models)"),
    ("5_visualization.py", "Visualization Suite"),
]

def run_pipeline():
    print("EMA STATISTICAL PIPELINE")
    start_time = time.time()
    
    for script, desc in stages:
        print(f"\n[+] Executing {script}: {desc}")
        try:
            result = subprocess.run(
                [sys.executable, script], 
                check=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"\n[!] PIPELINE HALTED: Error running {script}.")
            print("Please fix the error before continuing.")
            sys.exit(1)
            
    end_time = time.time()
    mins, secs = divmod(int(end_time - start_time), 60)
    print(f"  PIPELINE COMPLETELY FINISHED IN {mins}m {secs}s")
    print("  See model_comparison_report.txt for the final aggregated results.")


if __name__ == "__main__":
    run_pipeline()
