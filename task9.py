# task9_freeform.py
# Analysis: Factors driving loan offer acceptance.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV # For Section 6.1

# --- Configuration & Setup ---
BASE     = Path(__file__).resolve().parent
CSV_FILE = BASE / "Dhana-Loans-2025.csv"

SLA_MAP = {
    "car":                           28,
    "home improvement":             21,
    "loan takeover":                14,
    "existing loan takeover":       14,
}
DEFAULT_SLA = 28

# 1. LOAD & CLEAN
log = pd.read_csv(CSV_FILE, low_memory=False)
log["start_time"] = pd.to_datetime(log["start_time"], errors="coerce")
log["end_time"]   = pd.to_datetime(log["end_time"],   errors="coerce")

# Specifically handle OfferedAmount as it needs numeric conversion
for col in ["LoanGoal", "Activity", "LoanID", "OfferID", "Accepted", "Resource", "OfferedAmount"]:
    if col in log.columns:
        log[col] = log[col].astype(str).str.strip().str.lower()
        if col == "OfferedAmount":
            log[col] = pd.to_numeric(log[col], errors='coerce')

# 2. FEATURE ENGINEERING
def sla_days(goal: str) -> int:
    goal = (goal or "").lower()
    for k, v in SLA_MAP.items():
        if k in goal:
            return v
    return DEFAULT_SLA

gb = log.groupby("LoanID")

# Efficiently determine the primary loan goal for each application
goal_counts = (
    log
    .groupby(['LoanID','LoanGoal'])
    .size()
    .rename('cnt')
    .reset_index()
)
mode_df = (
    goal_counts
    .loc[ goal_counts.groupby('LoanID')['cnt'].idxmax() , ['LoanID','LoanGoal'] ]
    .rename(columns={'LoanGoal':'loan_goal'})
)

# Aggregate case-level information and merge primary loan goal
case_tbl = (
    gb
    .agg(
        credit_score = ("CreditScore",    "first"),
        requested_amt= ("RequestedAmount","first"),
        start         = ("start_time",    "min"),
        end           = ("end_time",      "max"),
    )
    .reset_index()
    .merge(mode_df, on="LoanID", how="left")
)
case_tbl['credit_score'] = pd.to_numeric(case_tbl['credit_score'], errors='coerce')
case_tbl['requested_amt'] = pd.to_numeric(case_tbl['requested_amt'], errors='coerce')

case_tbl["cycle_days"] = (case_tbl["end"] - case_tbl["start"]).dt.total_seconds() / 86400.0

# SLA violation
case_tbl["sla_days"]      = case_tbl["loan_goal"].apply(sla_days)
case_tbl["sla_violation"] = (case_tbl["cycle_days"] > case_tbl["sla_days"]).astype(int)

# Bank vs. customer time
milestones    = ["a_complete", "a_validating", "a_incomplete"]
milestone_log = log[log["Activity"].isin(milestones + ["a_create application"])]
tmp = milestone_log.sort_values(["LoanID","start_time"])
first_complete = tmp[tmp["Activity"]=="a_complete"].groupby("LoanID")["start_time"].first()
first_valid    = tmp[tmp["Activity"].isin(["a_validating","a_incomplete"])].groupby("LoanID")["start_time"].first()
case_tbl = (
    case_tbl
    .merge(first_complete.rename("t_complete"), on="LoanID", how="left")
    .merge(first_valid.rename("t_custresp"), on="LoanID", how="left")
)
case_tbl["bank_days"] = (case_tbl["t_complete"] - case_tbl["start"]).dt.total_seconds()/86400.0
case_tbl["cust_days"] = (case_tbl["t_custresp"] - case_tbl["t_complete"]).dt.total_seconds()/86400.0
case_tbl.drop(columns=["t_complete","t_custresp"], inplace=True)
case_tbl["bank_days"] = case_tbl["bank_days"].fillna(0)
case_tbl["cust_days"] = case_tbl["cust_days"].fillna(0)

# Target variable: offer accepted
accepted_ids = log[log["Accepted"]=="true"]["LoanID"].unique()
case_tbl["accepted"] = case_tbl["LoanID"].isin(accepted_ids).astype(int)

# Drop rows with NaN in critical columns after initial joins and calculations
case_tbl.dropna(subset=["loan_goal","credit_score","requested_amt"], inplace=True)

# Number of "A_Incomplete" events
n_inc = log[log["Activity"]=="a_incomplete"] \
            .groupby("LoanID").size() \
            .rename("n_incomplete")
case_tbl = case_tbl.merge(n_inc, on="LoanID", how="left")
case_tbl["n_incomplete"] = case_tbl["n_incomplete"].fillna(0)

# Number of unique offers created
n_off = log[log["Activity"]=="o_created"] \
            .groupby("LoanID")["OfferID"] \
            .nunique() \
            .rename("n_offers")
case_tbl = case_tbl.merge(n_off, on="LoanID", how="left")
case_tbl["n_offers"] = case_tbl["n_offers"].fillna(0)


# ADVANCED FEATURE ENGINEERING
case_tbl["cs_x_req_amt"] = case_tbl["credit_score"] * case_tbl["requested_amt"]
case_tbl["credit_score_sq"] = case_tbl["credit_score"] ** 2

if 'OfferedAmount' in log.columns and 'OfferID' in log.columns:
    offers_df = log.dropna(subset=['LoanID', 'OfferID', 'OfferedAmount']) \
                     .drop_duplicates(subset=['LoanID', 'OfferID'])
    sum_offered_amt = offers_df.groupby("LoanID")["OfferedAmount"].sum().rename("sum_offered_amt")
    case_tbl = case_tbl.merge(sum_offered_amt, on="LoanID", how="left")
    case_tbl["offered_to_req_ratio"] = case_tbl["sum_offered_amt"] / case_tbl["requested_amt"]
    case_tbl["offered_to_req_ratio"] = case_tbl["offered_to_req_ratio"].replace([np.inf, -np.inf], np.nan)
    case_tbl["sum_offered_amt"] = case_tbl["sum_offered_amt"].fillna(0)
    case_tbl["offered_to_req_ratio"] = case_tbl["offered_to_req_ratio"].fillna(0)
else:
    case_tbl["sum_offered_amt"] = 0
    case_tbl["offered_to_req_ratio"] = 0

if 'Resource' in log.columns and 'Activity' in log.columns:
    offer_creation_events = log[log['Activity'] == 'o_created']
    if not offer_creation_events.empty:
        n_unique_offer_resources = offer_creation_events.groupby('LoanID')['Resource'] \
                                       .nunique().rename('n_unique_offer_resources')
        case_tbl = case_tbl.merge(n_unique_offer_resources, on='LoanID', how='left')
        case_tbl['n_unique_offer_resources'] = case_tbl['n_unique_offer_resources'].fillna(0)
    else:
        case_tbl['n_unique_offer_resources'] = 0
else:
    case_tbl['n_unique_offer_resources'] = 0

# Group rare loan_goal categories into 'other'
min_count = 30
vc = case_tbl["loan_goal"].value_counts()
rare_cats = vc[vc < min_count].index
case_tbl.loc[case_tbl["loan_goal"].isin(rare_cats), "loan_goal"] = "other"


# 3. DESIGN MATRIX & TRAIN/TEST 
drop_cols = ["accepted", "LoanID", "start", "end", "sla_days"]
y = case_tbl["accepted"]
X_full = pd.get_dummies(case_tbl.drop(columns=drop_cols, errors='ignore'),
                        columns=["loan_goal"], drop_first=True)

loan_goal_dummy_cols = [col for col in X_full.columns if col.startswith("loan_goal_")]
if 'sla_violation' in X_full.columns:
    for goal_col in loan_goal_dummy_cols:
        interaction_col_name = f"sla_v_x_{goal_col.replace('loan_goal_', '')}"
        X_full[interaction_col_name] = X_full["sla_violation"] * X_full[goal_col]

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.25, random_state=42, stratify=y
)

potential_num_cols = [
    "credit_score", "requested_amt", "cycle_days",
    "bank_days", "cust_days", "n_incomplete", "n_offers",
    "cs_x_req_amt", "credit_score_sq",
    "sum_offered_amt", "offered_to_req_ratio",
    "n_unique_offer_resources"
]
num_cols = [col for col in potential_num_cols if col in X_train.columns]

if num_cols:
    scaler = StandardScaler().fit(X_train[num_cols])
    X_train[num_cols] = scaler.transform(X_train[num_cols])
    X_test [num_cols] = scaler.transform(X_test [num_cols])

all_cols = X_train.columns.union(X_test.columns)
X_train = X_train.reindex(columns=all_cols, fill_value=0)
X_test  = X_test .reindex(columns=all_cols, fill_value=0)

for df in (X_train, X_test):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

X_train_sm = sm.add_constant(X_train.copy(), has_constant="add") # Use a copy for statsmodels
X_test_sm  = sm.add_constant(X_test.copy() , has_constant="add") # Use a copy for statsmodels

zero_var_cols = X_train_sm.columns[(X_train_sm.var() == 0) & (X_train_sm.columns != "const")]
if not zero_var_cols.empty:
    X_train_sm = X_train_sm.drop(columns=zero_var_cols)
    X_test_sm  = X_test_sm.reindex(columns=X_train_sm.columns, fill_value=0)

mask = ~X_train_sm.isna().any(axis=1)
X_train_sm = X_train_sm.loc[mask]
y_train_sm = y_train.loc[mask] # y_train needs to be aligned with X_train_sm
X_test_sm = X_test_sm.reindex(columns=X_train_sm.columns, fill_value=0)

bad_cols = [
    "loan_goal_existing loan takeover",
    "loan_goal_home improvement",
]
X_train_sm = X_train_sm.drop(columns=bad_cols, errors="ignore")
X_test_sm  = X_test_sm.drop(columns=bad_cols, errors="ignore")
X_test_sm = X_test_sm.reindex(columns=X_train_sm.columns, fill_value=0)


# 4. STATSMODELS LOGIT MODEL FIT & OUTPUTS 
print(f"\nFitting statsmodels Logit model with {X_train_sm.shape[1]} features.")
model = sm.Logit(y_train_sm, X_train_sm.astype(float)).fit( # Use y_train_sm
            disp=False, maxiter=200)
print(model.summary(xname=X_train_sm.columns.tolist()))

odds_sm = np.exp(model.params).sort_values(ascending=False) # Renamed to odds_sm
print("\nTOP 15 ODDS RATIOS (statsmodels - full model):\n", odds_sm.head(15))

# 5. EVALUATION (Statsmodels)
preds_sm = model.predict(X_test_sm.astype(float)) # Renamed to preds_sm
print(f"\nStatsmodels ROC-AUC: {roc_auc_score(y_test, preds_sm):0.3f} | "
      f"Accuracy: {accuracy_score(y_test, (preds_sm > 0.5).astype(int)):0.3f}")

# ------------------------------------------------------------------
# 6. BAR CHART — friendly labels
# ------------------------------------------------------------------
def pretty_label(raw_name: str) -> str:
    """Convert internal feature names to readable labels for the plot."""
    if raw_name == "n_offers":
        return "# Offers"
    if raw_name == "sum_offered_amt":
        return "Total offered amount"
    if raw_name == "cs_x_req_amt":
        return "CreditScore × RequestedAmt"
    if raw_name == "credit_score_sq":
        return "CreditScore²"

    # Loan-goal dummies
    if raw_name.startswith("loan_goal_"):
        goal = raw_name.replace("loan_goal_", "").replace("_", " ")
        return f"Loan goal = {goal.title()}"

    # SLA × Loan-goal interactions
    if raw_name.startswith("sla_v_x_"):
        goal = raw_name.replace("sla_v_x_", "").replace("_", " ")
        return f"SLA × {goal.title()}"

    # Default: title-case the name and replace underscores
    return raw_name.replace("_", " ").title()

top_n = 15
odds_for_plot = odds_sm.drop(["const", "credit_score_sq"], errors="ignore")
# Select 15 largest by |log-odds| (same as |log OR|)
plot_idx = odds_for_plot.abs().nlargest(top_n).index
pretty_index = [pretty_label(c) for c in plot_idx]

plt.figure(figsize=(8, 7))
(odds_for_plot[plot_idx]
 .set_axis(pretty_index)
 .sort_values()
 .plot.barh())

plt.xlabel("Odds ratio (exp(β))")
plt.title(f"Top {top_n} Drivers (statsmodels)")
plt.tight_layout()
plt.savefig("task9_odds_pretty.png", dpi=300)
print("Chart saved: task9_odds_pretty.png")
plt.show()


# 6.1 ALTERNATIVE ODDS RATIO VISUALIZATION (Sklearn Model without CreditScoreSq)

print("\n--- Alternative Odds Ratios: Sklearn Model (excluding credit_score_sq from model) ---")
X_train_sklearn_alt = X_train.copy() # Original scaled X_train
X_test_sklearn_alt = X_test.copy()   # Original scaled X_test

feature_to_exclude = 'credit_score_sq'
if feature_to_exclude in X_train_sklearn_alt.columns:
    X_train_sklearn_alt = X_train_sklearn_alt.drop(columns=[feature_to_exclude])
    X_test_sklearn_alt  = X_test_sklearn_alt.drop(columns=[feature_to_exclude], errors='ignore')
    # Ensure columns match after drop, in case feature wasn't in X_test for some reason
    X_test_sklearn_alt = X_test_sklearn_alt.reindex(columns=X_train_sklearn_alt.columns, fill_value=0)
    print(f"   (For this Sklearn model, '{feature_to_exclude}' was excluded before training)")


# 8. DECILE‐BASED LIFT CHART & GAIN TABLE
print("\n--- Decile Lift Analysis (based on main statsmodels predictions) ---")
df_lift = pd.DataFrame({'y_true': y_test, 'y_prob': preds_sm}) # Use main statsmodels preds_sm
df_lift['decile'] = pd.qcut(df_lift['y_prob'], 10, labels=False, duplicates='drop')
overall_rate = df_lift['y_true'].mean()
lift_table = (
    df_lift.groupby('decile')
    .agg(total=('y_true','size'), accepted=('y_true','sum'))
    .assign(
        acceptance_rate=lambda d: d['accepted']/d['total'],
        lift=lambda d: d['acceptance_rate']/overall_rate
    ).sort_index(ascending=False)
)
print("\nDECILE LIFT TABLE:")
print(lift_table.round(3))

plt.figure(figsize=(7,4))
lift_table['lift'].plot.barh()
plt.xlabel("Lift over average acceptance rate")
plt.ylabel("Decile (9 = top 10%)")
plt.title("Lift by Predicted‐Probability Decile (Statsmodels)")
plt.tight_layout()
plt.savefig("task9_lift_chart.png", dpi=300)
print("Chart saved: task9_lift_chart.png")
plt.show()


# 9. SEGMENTED ANALYSIS OF SLA VIOLATION

def plot_segmented_sla_effect(
    df,
    segment_col,
    x_label,
    title,
    filename_key,
    figsize=(12, 7),
):
    """
    Groups data by a segment column and SLA violation, then plots acceptance rates.
    """
    print(f"\nAnalyzing Acceptance Rate by {x_label} and SLA Violation:")
    try:
        # Ensure 'accepted' and 'sla_violation' are present
        if not all(col in df.columns for col in [segment_col, "sla_violation", "accepted"]):
            print(f"Missing required columns for segment {segment_col}. Skipping.")
            return

        grouped_data = (
            df.groupby([segment_col, "sla_violation"])["accepted"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "acceptance_rate"})
        )
        print(grouped_data)

        # Check if there's anything to plot (e.g., after unstacking)
        unstacked_data = grouped_data["acceptance_rate"].unstack(
            level="sla_violation"
        )
        if unstacked_data.empty or unstacked_data.isnull().all().all():
            print(f"No data to plot for {x_label} after unstacking. Skipping plot.")
            return

        fig, ax = plt.subplots(figsize=figsize)
        unstacked_data.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Acceptance Rate")
        ax.set_xlabel(x_label)
        plt.xticks(rotation=45, ha="right")
        ax.legend(title="SLA Violated (0=No, 1=Yes)")
        plt.tight_layout()
        filepath = f"task9_sla_effect_{filename_key}.png"
        plt.savefig(filepath, dpi=300)
        print(f"Chart saved: {filepath}")
        plt.show()
    except (TypeError, KeyError, ValueError) as e: # Catch common pandas/plotting errors
        print(f"Could not plot SLA effect for {x_label}: {e}")
    except Exception as e: # General fallback for unexpected issues
        print(f"An unexpected error occurred while plotting for {x_label}: {e}")


print("\n--- Segmented Analysis of SLA Violation (using case_tbl) ---")
required_cols_for_sla_analysis = ["sla_violation", "loan_goal", "accepted"]
if all(col in case_tbl.columns for col in required_cols_for_sla_analysis):
    # Analysis by Loan Goal
    plot_segmented_sla_effect(
        df=case_tbl,
        segment_col="loan_goal",
        x_label="Loan Goal",
        title="Acceptance Rate by Loan Goal and SLA Violation",
        filename_key="by_goal",
    )

    # Analysis by Requested Amount Quantile
    if "requested_amt" in case_tbl.columns:
        # Create a temporary copy for this specific analysis to add quantile column
        # and handle numeric conversion/NaNs without affecting original case_tbl elsewhere.
        temp_df_for_amt_analysis = case_tbl.copy()
        temp_df_for_amt_analysis["requested_amt"] = pd.to_numeric(
            temp_df_for_amt_analysis["requested_amt"], errors="coerce"
        )
        temp_df_for_amt_analysis.dropna(subset=["requested_amt"], inplace=True)

        if temp_df_for_amt_analysis["requested_amt"].nunique() > 1:
            try:
                temp_df_for_amt_analysis["req_amt_quantile"] = pd.qcut(
                    temp_df_for_amt_analysis["requested_amt"],
                    q=4,
                    labels=False,
                    duplicates="drop",
                )
                plot_segmented_sla_effect(
                    df=temp_df_for_amt_analysis,
                    segment_col="req_amt_quantile",
                    x_label="Requested Amount Quantile (0=lowest)",
                    title="Acceptance Rate by Requested Amount Quantile & SLA Violation",
                    filename_key="by_req_amt",
                    figsize=(10, 6),
                )
            except ValueError as e: # Specific to qcut
                print(f"Could not create quantiles for requested_amt: {e}")
        else:
            print(
                "Skipping SLA analysis by requested_amt: Not enough unique "
                "values after cleaning for quantiles."
            )
    else:
        print(
            "Skipping SLA analysis by requested_amt: 'requested_amt' column missing."
        )
else:
    missing = [col for col in required_cols_for_sla_analysis if col not in case_tbl.columns]
    print(f"Skipping segmented SLA analysis: Key column(s) missing: {missing}")

print("\n--- Script execution finished ---")

