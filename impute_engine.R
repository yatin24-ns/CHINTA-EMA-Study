# impute_engine.R
# This script is called by 1_organisation.py as a decoupled R-Py bridge.
# It performs True Multilevel Imputation (FCS-LMM) using mice 2l.pan.
#
# MIDO Strategy: S2_Fatigue_Mean (the outcome variable) is NEVER imputed.
#   It is included only as an auxiliary predictor in the imputation model to
#   preserve the covariance structure (congeniality). Python restores the
#   original observed/missing pattern for the outcome after reading this output.
#
# Imputation paths:
#   Both JAV and Passive imputation paths are handled by Python post-hoc.
#   This script imputes: S1_Workload_Mean, S1_Workload_SD, Leisure_Log,
#   Sleep_Quality, Interaction_Normal, Interaction_Crunch.
#   (Interaction variables are pre-calculated in Python for the JAV path,
#    so they are present in the input and can be imputed directly here.)

# Suppress warnings for cleaner output
options(warn=-1)

# Check for required packages and prompt if missing
packages <- c("mice", "pan", "arrow", "dplyr")
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste("Package", pkg, "is required but not installed. Please install it using install.packages()"))
  }
}

suppressPackageStartupMessages(library(mice))
suppressPackageStartupMessages(library(pan))
suppressPackageStartupMessages(library(arrow))
suppressPackageStartupMessages(library(dplyr))

# File paths (relative to working directory set by 1_organisation.py)
input_file  <- "Formatted_Data/temp_for_imputation.parquet"
output_file <- "Formatted_Data/imputed_m5_datasets.parquet"

if (!file.exists(input_file)) {
  stop("Input file not found: ", input_file)
}

cat("[R] Loading data for imputation...\n")
df <- read_parquet(input_file)

# Ensure Participant_No is integer for 2l.pan clustering
df$participant_numeric <- as.integer(as.factor(df$Participant_No))

# ---------------------------------------------------------------------------
# MIDO: Impute the outcome variable S2_Fatigue_Mean to allow it to be used
# as a predictor for other missing variables, but Python will discard these
# imputed values later (Von Hippel's Impute, Then Delete).
# ---------------------------------------------------------------------------
cols_to_impute <- c(
  "S2_Fatigue_Mean",
  "S1_Workload_Mean",
  "S1_Workload_SD",
  "Leisure_Log",
  "Sleep_Quality",
  "Interaction_Normal",
  "Interaction_Crunch"
)

# Ensure all imputation targets are numeric
for (col in cols_to_impute) {
  if (col %in% colnames(df)) {
    df[[col]] <- as.numeric(df[[col]])
  }
}
# Also ensure S2_Fatigue_Mean is numeric so it works as an auxiliary predictor
if ("S2_Fatigue_Mean" %in% colnames(df)) {
  df[["S2_Fatigue_Mean"]] <- as.numeric(df[["S2_Fatigue_Mean"]])
}

# Define the predictor matrix
cat("[R] Building predictor matrix (MIDO strategy)...\n")
ini  <- mice(df, maxit = 0, print = FALSE)
pred <- ini$predictorMatrix

# Initialise all to 0 — build a minimal predictor matrix to prevent
# 2l.pan from crashing on sparse data with too many predictors
pred[,] <- 0

# Auxiliary predictors used for each imputed variable
# S2_Fatigue_Mean is listed here ONLY as a predictor (row left at 0 → not imputed)
valid_preds <- c(
  "day_number", "S2_Fatigue_Mean", "Trait_Fatigue",
  "S1_Workload_Mean", "S1_Workload_SD",
  "Leisure_Log", "Sleep_Quality",
  "Interaction_Normal", "Interaction_Crunch"
)

for (col in cols_to_impute) {
  # Cluster variable for 2l.pan
  pred[col, "participant_numeric"] <- -2

  for (vp in valid_preds) {
    if (vp %in% colnames(pred) && vp != col) {
      pred[col, vp] <- 1
    }
  }
}

# Set imputation method: 2l.pan for all targets, "" for everything else
meth      <- ini$method
meth[]    <- ""
for (col in cols_to_impute) {
  if (col %in% names(meth)) {
    meth[col] <- "2l.pan"
  }
}

cat("[R] Predictor matrix row for S1_Workload_Mean:\n")
print(pred["S1_Workload_Mean", ])

cat(sprintf("[R] Running FCS-LMM Multilevel Imputation (m=20, maxit=20, seed=42)...\n"))
imp <- mice(df,
            method          = meth,
            predictorMatrix = pred,
            m               = 20,
            maxit           = 20,
            seed            = 42,
            print           = FALSE)

cat("[R] Imputation complete. Extracting m=20 datasets...\n")
long_imp <- complete(imp, action = "long", include = FALSE)

cat("[R] Saving imputed datasets to: ", output_file, "\n")
write_parquet(long_imp, output_file)
cat("[R] Done. Exiting.\n")
