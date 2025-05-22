import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Dhana-Loans-2025.csv', parse_dates=['start_time', 'end_time'])
df['Activity'] = df['Activity'].astype(str).str.strip()
df['LoanID'] = df['LoanID'].astype(str).str.strip()

# Sort by LoanID and end_time to ensure correct order of activities
df = df.sort_values(['LoanID', 'end_time'])

# 1. Count number of incompleteness events per application
incomplete_counts = (
    df[df['Activity'] == 'A_Incomplete']
    .groupby('LoanID')
    .size()
    .reset_index(name='incomplete_count')
)

# 2. Determine if the final activity is 'A_Pending' (application accepted)
final_activity = (
    df.groupby('LoanID')['Activity']
    .agg(lambda x: x.iloc[-1])
    .reset_index(name='final_activity')
)
final_activity['is_accepted'] = final_activity['final_activity'] == 'A_Pending'

# 3. Merge and analyze
analysis = pd.merge(final_activity, incomplete_counts, on='LoanID', how='left').fillna(0)

# 4. Group by incompleteness count and calculate acceptance rate
acceptance_by_incompleteness = (
    analysis.groupby('incomplete_count')
    .agg(
        total_cases=('LoanID', 'count'),
        accepted_cases=('is_accepted', 'sum')
    )
    .reset_index()
)
acceptance_by_incompleteness['acceptance_rate'] = (
    acceptance_by_incompleteness['accepted_cases'] / acceptance_by_incompleteness['total_cases'] * 100
)

print("\nAcceptance rates by number of incompleteness events:")
print(acceptance_by_incompleteness.to_string(index=False, float_format='%.2f'))

# 5. Additional: Average incompleteness for accepted vs not accepted
avg_incomplete_by_outcome = (
    analysis.groupby('is_accepted')
    .agg(
        avg_incomplete_count=('incomplete_count', 'mean'),
        total_cases=('LoanID', 'count')
    )
    .reset_index()
)
print("\nAverage number of incompleteness events by final acceptance:")
print(avg_incomplete_by_outcome.to_string(index=False, float_format='%.2f'))