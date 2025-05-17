import pandas as pd

df = pd.read_csv('Dhana-Loans-2025.csv', parse_dates=['start_time', 'end_time'])
df['Activity'] = df['Activity'].astype(str).str.strip()
df['LoanID'] = df['LoanID'].astype(str).str.strip()

# Count activity occurrences per case by definition froma task
activity_counts = (
    df.groupby(['LoanID', 'Activity'])
    .size()
    .reset_index(name='count')
)

rework_events = activity_counts[activity_counts['count'] > 1]

# flag each case if at least any rework
loanids_with_rework = rework_events['LoanID'].unique()
df['HasRework'] = df['LoanID'].isin(loanids_with_rework)

# cycle time agg for rework cases
case_durations = (
    df.groupby('LoanID')
    .agg(
        start=('start_time', 'min'),
        end=('end_time', 'max'),
        has_rework=('HasRework', 'max')
    )
    .reset_index()
)

case_durations['start'] = pd.to_datetime(case_durations['start'], format='ISO8601')
case_durations['end'] = pd.to_datetime(case_durations['end'], format='ISO8601')

case_durations['cycle_time_days'] = (case_durations['end'] - case_durations['start']).dt.total_seconds() / 86400

# comparison rework vs no rework
avg_duration_with_rework = case_durations[case_durations['has_rework']]['cycle_time_days'].mean()
avg_duration_without_rework = case_durations[~case_durations['has_rework']]['cycle_time_days'].mean()
rework_impact = avg_duration_with_rework - avg_duration_without_rework

print(f"Cases with rework: {case_durations['has_rework'].sum()}")
print(f"Cases without rework: {len(case_durations) - case_durations['has_rework'].sum()}")
print(f"Average CT without rework: {avg_duration_without_rework:.2f} days")
print(f"Average CT with rework: {avg_duration_with_rework:.2f} days")
print(f"CT increase due to rework: {rework_impact:.2f} days")

# Cases with rework: 10360
# Cases without rework: 9933
# Average CT without rework: 21.83 days
# Average CT with rework: 22.10 days
# CT increase due to rework: 0.27 days

rework_summary = (
    rework_events.groupby('Activity')
    .agg(num_cases=('LoanID', 'nunique'), total_rework_occurrences=('count', 'sum'))
    .reset_index()
    .sort_values(by='num_cases', ascending=False)
)
print("\nRework activities:")
print(rework_summary.to_string(index=False))

# Rework activities:
#                 Activity  num_cases  total_rework_occurrences
#   W_Validate application       7286                     17593
#             A_Validating       7202                     17250
#                O_Created       5197                     11660
#           O_Create Offer       5197                     11660
# O_Sent (mail and online)       4263                      9415
#  W_Call incomplete files       3433                      8019
#             A_Incomplete       3404                      7937
#              O_Cancelled       2226                      5042
#               O_Returned        710                      1429
#                O_Refused        482                      1062
#   W_Complete application         91                       189
#     O_Sent (online only)         45                       106
#      W_Call after offers         40                        81
#           W_Handle leads         24                        53
# W_Assess potential fraud         21                        45