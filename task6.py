import pandas as pd

# Load the dataset
df = pd.read_csv('Dhana-Loans-2025.csv', parse_dates=['start_time', 'end_time'])
df['Activity'] = df['Activity'].astype(str).str.strip()
df['LoanID'] = df['LoanID'].astype(str).str.strip()

# Flag cases with fraud assessment
fraud_cases = df[df['Activity'] == 'W_Assess potential fraud']['LoanID'].unique()
df['FraudAssessment'] = df['LoanID'].isin(fraud_cases)

# Calculate cycle time for each case (from first to last event)
cycle_times = (
    df.groupby('LoanID')
    .agg(
        start_time=('start_time', 'min'),
        end_time=('end_time', 'max'),
        fraud_assessment=('FraudAssessment', 'max')
    )
    .reset_index()
)

cycle_times['start_time'] = pd.to_datetime(cycle_times['start_time'], format='ISO8601')
cycle_times['end_time'] = pd.to_datetime(cycle_times['end_time'], format='ISO8601')

cycle_times['cycle_time_days'] = (cycle_times['end_time'] - cycle_times['start_time']).dt.total_seconds() / 86400

# Compare average cycle time
avg_cycle_fraud = cycle_times[cycle_times['fraud_assessment']]['cycle_time_days'].mean()
avg_cycle_no_fraud = cycle_times[~cycle_times['fraud_assessment']]['cycle_time_days'].mean()

print(f"Average cycle time (with fraud assessment): {avg_cycle_fraud:.2f} days")
print(f"Average cycle time (without fraud assessment): {avg_cycle_no_fraud:.2f} days") 