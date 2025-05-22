import pandas as pd

# Load the dataset
df = pd.read_csv('Dhana-Loans-2025.csv', parse_dates=['start_time', 'end_time'])
df['Activity'] = df['Activity'].astype(str).str.strip()
df['LoanID'] = df['LoanID'].astype(str).str.strip()
df['OfferID'] = df['OfferID'].astype(str).str.strip()
df['Accepted'] = df['Accepted'].astype(str).str.strip().str.lower()

# Filter to offer creation events
offers = df[df['Activity'] == 'O_Created']
# Count unique offers per customer (LoanID)
offer_counts = offers.groupby('LoanID')['OfferID'].nunique().reset_index(name='num_offers')
# How many customers received more than one offer?
more_than_one_offer = offer_counts[offer_counts['num_offers'] > 1]
num_customers_more_than_one = more_than_one_offer.shape[0]
total_customers = offer_counts.shape[0]
percentage = num_customers_more_than_one / total_customers * 100
print(f"Number of customers who received more than one offer: {num_customers_more_than_one}")
print(f"Total number of customers: {total_customers}")
print(f"Percentage of customers with more than one offer: {percentage:.2f}%\n")

# Define application state changes (all 'A_' activities)
application_states = [a for a in df['Activity'].unique() if a.startswith('A_')]
results = []
for loan_id, group in df[df['LoanID'].notnull()].groupby('LoanID'):
    group = group.sort_values('start_time')
    # Find A_Accepted
    accepted = group[group['Activity'] == 'A_Accepted']
    if accepted.empty:
        continue
    accepted_time = accepted['start_time'].iloc[0]
    # Find the next application state change after A_Accepted
    after_accepted = group[group['start_time'] > accepted_time]
    next_state = after_accepted[after_accepted['Activity'].isin(application_states)]
    if not next_state.empty:
        next_state_time = next_state['start_time'].iloc[0]
    else:
        next_state_time = group['start_time'].max()
    # Count unique offers created between A_Accepted and next application state
    offers_in_window = group[
        (group['Activity'] == 'O_Created') &
        (group['start_time'] >= accepted_time) &
        (group['start_time'] < next_state_time)
    ]['OfferID'].nunique()
    results.append({
        'LoanID': loan_id,
        'offers_in_conversation': offers_in_window
    })
results_df = pd.DataFrame(results)
multi_offer_conversations = results_df[results_df['offers_in_conversation'] > 1]
print(f"Number of applications with multiple offers in the same conversation: {multi_offer_conversations.shape[0]}")
print(f"Percentage of such applications: {100 * multi_offer_conversations.shape[0] / results_df.shape[0]:.2f}%")

# For each LoanID, check if any offer was accepted (Accepted == 'true')
offer_accepted = (
    df.groupby('LoanID')['Accepted']
    .apply(lambda x: 'true' in x.values)
    .reset_index(name='offer_accepted')
)
# Merge with offer_counts
offer_accept_analysis = pd.merge(offer_counts, offer_accepted, on='LoanID', how='left')
# Group by number of offers and calculate acceptance rate
acceptance_by_num_offers = (
    offer_accept_analysis.groupby('num_offers')
    .agg(
        total_cases=('LoanID', 'count'),
        accepted_cases=('offer_accepted', 'sum')
    )
    .reset_index()
)
acceptance_by_num_offers['acceptance_rate'] = (
    acceptance_by_num_offers['accepted_cases'] / acceptance_by_num_offers['total_cases'] * 100
)
print("Acceptance rate by number of offers per case:")
print(acceptance_by_num_offers.to_string(index=False, float_format='%.2f')) 