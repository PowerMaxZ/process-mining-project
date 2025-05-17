import pandas as pd
import numpy as np

# TASK 1 

df = pd.read_csv('Dhana-Loans-2025.csv', parse_dates=['start_time', 'end_time'])

# saw there could be trailing spaces in the LoanGoal column
df['LoanGoal'] = df['LoanGoal'].astype(str).str.strip().str.lower()

sla_mapping = {
    'car': 28,
    'home improvement': 21,
    'loan takeover': 14,
    'existing loan takeover': 14,
}

# Default SLA for any 'other' type
default_sla = 28

# return SLA days based on LoanGoal
def get_sla_days(loan_goal):
    loan_goal_clean = loan_goal.strip().lower()
    for key in sla_mapping:
        if key in loan_goal_clean:
            return sla_mapping[key]
    return default_sla

# loanID is basically a caseid so we use it to group to beginning and end times of each case
process_times = (
    df.groupby('LoanID')
      .agg(
          ApplicationType=('ApplicationType', 'first'),
          LoanGoal=('LoanGoal', 'first'),
          RequestedAmount=('RequestedAmount', 'first'),
          start_time=('start_time', 'min'),
          end_time=('end_time', 'max')
      )
      .reset_index()
)

process_times['start_time'] = pd.to_datetime(process_times['start_time'], format='ISO8601')
process_times['end_time'] = pd.to_datetime(process_times['end_time'], format='ISO8601')

process_times['process_duration_days'] = (process_times['end_time'] - process_times['start_time']).dt.days

process_times['sla_days'] = process_times['LoanGoal'].apply(get_sla_days)

# new column to mark if SLA was violated, then used to calculate statistics
process_times['SLA_Violated'] = process_times['process_duration_days'] > process_times['sla_days']

# statistics
report = (
    process_times.groupby('LoanGoal')
        .agg(
            total_apps=('LoanID', 'count'),
            violations=('SLA_Violated', 'sum'),
            sla_days=('sla_days', 'max')
        )
        .reset_index()
)

report['violation_rate_percent'] = 100 * report['violations'] / report['total_apps']

print("SLA Compliance Report per LoanGoal:")
print(report.to_string(index=False))



# SLA Compliance Report per LoanGoal:
#               LoanGoal  total_apps  violations  sla_days  violation_rate_percent
#                   boat         166          58        28               34.939759
#          business goal          24          11        28               45.833333
#                    car        7680        2737        28               35.638021
#       caravan / camper         299          90        28               30.100334
#     debt restructuring           2           2        28              100.000000
# existing loan takeover        4793        3149        14               65.699979
#   extra spending limit         408         144        28               35.294118
#       home improvement        5898        2700        21               45.778230
#             motorcycle         227          81        28               35.682819
#    remaining debt home         680         321        28               47.205882
#           tax payments         116          40        28               34.482759