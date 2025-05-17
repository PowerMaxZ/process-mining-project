import pandas as pd

# question is a bit open-ended, so I have tried to cover as much info as possible

df = pd.read_csv('Dhana-Loans-2025.csv', parse_dates=['start_time', 'end_time'])
df['Activity'] = df['Activity'].astype(str).str.strip()
df['LoanID'] = df['LoanID'].astype(str).str.strip()

# first find cancelled applications
cancel_activities = ["A_Cancelled", "O_Cancelled"]
cancel_events = df[df['Activity'].isin(cancel_activities)]
cancelled_loanids = cancel_events['LoanID'].unique()

print(f"Total cancelled cases: {len(cancelled_loanids)}")
# Total cancelled cases: 10084

# flag cancelled ones for easier analysis
df['IsCancelled'] = df['LoanID'].isin(cancelled_loanids)

# not statistics
cancelled_cases = (
    df[df['IsCancelled']]
    .groupby('LoanID')
    .agg(
        ApplicationType=('ApplicationType', 'first'),
        LoanGoal=('LoanGoal', 'first'),
        RequestedAmount=('RequestedAmount', 'first'),
        CreditScore=('CreditScore', 'first')
    )
    .reset_index()
)

print("\nSummary statistics for cancelled loans:")
print(cancelled_cases.describe(include='all'))
# Summary statistics for cancelled loans:
#                        LoanID ApplicationType LoanGoal  RequestedAmount   CreditScore
# count                   10084           10084    10084     10084.000000  10084.000000
# unique                  10084               1       11              NaN           NaN
# top     Application_999544538      New credit      Car              NaN           NaN
# freq                        1           10084     3985              NaN           NaN
# mean                      NaN             NaN      NaN     16609.756248     55.097580
# std                       NaN             NaN      NaN     13879.731160    216.682774
# min                       NaN             NaN      NaN         0.000000      0.000000
# 25%                       NaN             NaN      NaN      7000.000000      0.000000
# 50%                       NaN             NaN      NaN     13000.000000      0.000000
# 75%                       NaN             NaN      NaN     20712.500000      0.000000
# max                       NaN             NaN      NaN    400000.000000   1127.000000

print("\nMost common ApplicationType for cancelled loans:")
print(cancelled_cases['ApplicationType'].value_counts().head())
# Most common ApplicationType for cancelled loans:
# ApplicationType
# New credit    10084
# Name: count, dtype: int64

print("\nMost common LoanGoal for cancelled loans:")
print(cancelled_cases['LoanGoal'].value_counts().head())
# Most common LoanGoal for cancelled loans:
# LoanGoal
# Car                       3985
# Home improvement          2788
# Existing loan takeover    2305
# Remaining debt home        382
# Extra spending limit       188
# Name: count, dtype: int64

# comparis with non-cancelled ones
not_cancelled_cases = (
    df[~df['IsCancelled']]
    .groupby('LoanID')
    .agg(
        ApplicationType=('ApplicationType', 'first'),
        LoanGoal=('LoanGoal', 'first'),
        RequestedAmount=('RequestedAmount', 'first'),
        CreditScore=('CreditScore', 'first')
    )
    .reset_index()
)

print("\nMean requested amount (cancelled):", cancelled_cases['RequestedAmount'].astype(float).mean())
print("Mean requested amount (not cancelled):", not_cancelled_cases['RequestedAmount'].astype(float).mean())

# Mean requested amount (cancelled): 16609.756247520825
# Mean requested amount (not cancelled): 16244.002448819669


print("\nMean credit score (cancelled):", pd.to_numeric(cancelled_cases['CreditScore'], errors='coerce').mean())
print("Mean credit score (not cancelled):", pd.to_numeric(not_cancelled_cases['CreditScore'], errors='coerce').mean())

# Mean credit score (cancelled): 55.09758032526775
# Mean credit score (not cancelled): 689.07973356842

print("\nTop 5 LoanGoals among cancelled applications:")
print(cancelled_cases['LoanGoal'].value_counts().head())
# Top 5 LoanGoals among cancelled applications:
# LoanGoal
# Car                       3985
# Home improvement          2788
# Existing loan takeover    2305
# Remaining debt home        382
# Extra spending limit       188
# Name: count, dtype: int64

print("\nTop 5 ApplicationTypes among cancelled applications:")
print(cancelled_cases['ApplicationType'].value_counts().head())
# Top 5 ApplicationTypes among cancelled applications:
# ApplicationType
# New credit    10084
# Name: count, dtype: int64

# cancellation per resource
cancel_by_resource = (
    cancel_events['Resource'].value_counts()
    .reset_index()
)
cancel_by_resource.columns = ['Resource', 'CancellationCount']

# total loans per resource
handled_by_resource = (
    df.groupby('Resource')['LoanID'].nunique()
    .reset_index()
    .rename(columns={'LoanID': 'TotalLoansHandled'})
)

# cancellation rate
cancel_by_resource = pd.merge(cancel_by_resource, handled_by_resource, on='Resource', how='left')
cancel_by_resource['CancellationRate'] = (
    cancel_by_resource['CancellationCount'] / cancel_by_resource['TotalLoansHandled']
)

print("\nEmployees with highest cancellation rates (min 10 loans handled):")
print(cancel_by_resource[cancel_by_resource['TotalLoansHandled'] >= 10]
      .sort_values('CancellationRate', ascending=False).head(10))

# Employees with highest cancellation rates (min 10 loans handled):
#     Resource  CancellationCount  TotalLoansHandled  CancellationRate
# 0     User_1              12205              16733          0.729397
# 60   User_20                 49                132          0.371212
# 58  User_128                 51                148          0.344595
# 85    User_9                 22                 74          0.297297
# 28   User_43                 95                356          0.266854
# 27  User_107                 98                430          0.227907
# 21  User_115                118                557          0.211849
# 44  User_129                 63                321          0.196262
# 6   User_102                184                956          0.192469
# 1    User_29                328               1712          0.191589

# Okay, so mean loan amount is almost the same, but credit score is significantly lower for cancelled applications
# Also, the most common loan goal is car
# and the most common application type is new credit
# new credit is the only application type cancelled but I think it is only because it is the only one present in the dataset
# User 1 has the highest cancellation rate, but also the most loans handled, 72% of loans handled by User 1 are cancelled, so maybe bank has to take a look at this user :D