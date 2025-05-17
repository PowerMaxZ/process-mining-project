import pandas as pd

milestones = [
    "A_Create Application",
    "A_Submitted",
    "A_Concept",
    "A_Accepted",
    "A_Complete",
    "A_Validating",
    "A_Incomplete",
    "A_Pending",
    "A_Denied",
    "A_Cancelled"
]

df = pd.read_csv('Dhana-Loans-2025.csv', parse_dates=['start_time', 'end_time'])
df['Activity'] = df['Activity'].astype(str).str.strip()
df['LoanID'] = df['LoanID'].astype(str).str.strip()

# taking first appearance of each milestone for each loan
milestone_events = (
    df[df['Activity'].isin(milestones)]
    .sort_values(['LoanID', 'start_time'])
    .groupby(['LoanID', 'Activity'], as_index=False)
    .first()[['LoanID', 'Activity', 'start_time']]
)

# pivot table for easier processing with each milestone as a column
milestone_times = milestone_events.pivot(index='LoanID', columns='Activity', values='start_time').reset_index()

# taking case start time  for each loan
first_event = (
    df.groupby('LoanID')['start_time'].min()
    .rename('ProcessStart')
    .reset_index()
)
milestone_times = pd.merge(milestone_times, first_event, on='LoanID', how='left')

# had an error, apparently after pivoting some columns could be strings, so back to dt again
for col in ['ProcessStart'] + milestones:
    if col in milestone_times.columns:
        milestone_times[col] = milestone_times[col].apply(pd.to_datetime)

# for each milestone we are calculating the time from case start to that milestone
for milestone in milestones:
    if milestone in milestone_times.columns:
        milestone_times[f'cycle_to_{milestone}'] = (
            (milestone_times[milestone] - milestone_times['ProcessStart']).dt.total_seconds() / 86400
        )
    else:
        milestone_times[f'cycle_to_{milestone}'] = float('nan')

# now average it
cycle_summary = []
for milestone in milestones:
    avg_days = milestone_times[f'cycle_to_{milestone}'].mean()
    cycle_summary.append({'Milestone': milestone, 'AverageCycleTime_Days': avg_days})
summary_df = pd.DataFrame(cycle_summary)

print("Average cycle time to reach each milestone (in days):")
print(summary_df.to_string(index=False, float_format='%.2f'))

# Average cycle time to reach each milestone (in days):
#            Milestone  AverageCycleTime_Days
# A_Create Application                   0.00
#          A_Submitted                   0.00
#            A_Concept                   0.04
#           A_Accepted                   1.61
#           A_Complete                   1.70
#         A_Validating                  11.44
#         A_Incomplete                  13.39
#            A_Pending                  18.15
#             A_Denied                  16.29
#          A_Cancelled                  29.90

# now bank vs customer 
# was no info about bank/customer time in task description, so I made simple assumption:
# Bank time is from first application event to sending out offers/requests (A_Complete).
# Customer time is from when the bank sends the offer (A_Complete) to when the customer reacts (first of A_Validating or A_Incomplete).

milestone_times['bank_time'] = (
    (milestone_times['A_Complete'] - milestone_times['ProcessStart']).dt.total_seconds() / 86400
)

milestone_times['customer_time'] = float('nan')
for idx, row in milestone_times.iterrows():
    complete = row['A_Complete']
    if pd.notnull(complete):
        customer_steps = []
        for cm in ['A_Validating', 'A_Incomplete']:
            t = row.get(cm, None)
            if pd.notnull(t) and t > complete:
                customer_steps.append((t - complete).total_seconds() / 86400)
        if customer_steps:
            milestone_times.at[idx, 'customer_time'] = min(customer_steps)
        else:
            milestone_times.at[idx, 'customer_time'] = float('nan')
    else:
        milestone_times.at[idx, 'customer_time'] = float('nan')

avg_bank_time = pd.to_numeric(milestone_times['bank_time'], errors='coerce').mean()
avg_customer_time = pd.to_numeric(milestone_times['customer_time'], errors='coerce').mean()

print("\nAverage time spent in bank activities (days): %.2f" % avg_bank_time)
print("Average time spent waiting for customer (days): %.2f" % avg_customer_time)

# Average time spent in bank activities (days): 1.70
# Average time spent waiting for customer (days): 9.82
