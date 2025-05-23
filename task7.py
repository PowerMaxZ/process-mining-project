import pandas as pd

# Load the dataset
df = pd.read_csv('Dhana-Loans-2025.csv', parse_dates=['start_time', 'end_time'])
df['Activity'] = df['Activity'].astype(str).str.strip()
df['LoanID'] = df['LoanID'].astype(str).str.strip()
df['OfferID'] = df['OfferID'].astype(str).str.strip()
df['Accepted'] = df['Accepted'].astype(str).str.strip().str.lower()

# 1. Retrieving how many customers received more than one offer. 
offers = df[df['Activity'] == 'O_Created']
offer_counts = offers.groupby('LoanID')['OfferID'].nunique().reset_index(name='num_offers')
num_customers_more_than_one = (offer_counts['num_offers'] > 1).sum()
total_customers = offer_counts.shape[0]

print(f"Number of customers who received more than one offer: {num_customers_more_than_one}")
print(f"Total number of customers: {total_customers}")
print(f"Percentage of customers with more than one offer: {100 * num_customers_more_than_one / total_customers:.2f}%\n")


# 2. Count how many applications have multiple offers in a single conversation (application). 
results = []
for loan_id, group in df.groupby('LoanID'):
    group = group.sort_values('end_time')
    activities = group['Activity'].tolist()
    try:
        idx_accept = activities.index('A_Accepted')
        idx_complete = activities.index('A_Complete', idx_accept + 1)
    except ValueError:
        continue

    # Extract events that happened during the conversation window
    conversation_start = idx_accept + 1
    conversation_end = idx_complete
    conversation_events = group.iloc[conversation_start:conversation_end]
    
    # Filter for only offer creation events during the conversation
    offer_creation_events = conversation_events[conversation_events['Activity'] == 'O_Created']
    
    # Count unique offers created during this conversation window
    unique_offers_in_conversation = offer_creation_events['OfferID'].nunique()
    results.append(unique_offers_in_conversation)

total_valid_applications = len(results)
num_multi_offer_conversations = sum(offer_count > 1 for offer_count in results)
print(f"Number of applications with multiple offers in the same conversation: {num_multi_offer_conversations}")
print(f"Total valid applications analyzed: {total_valid_applications}")
print(f"Percentage of such applications: {100 * num_multi_offer_conversations / total_valid_applications:.2f}%\n")


# 3. Does the number of offers impact offer acceptance?

# Check if each customer accepted any offer
offer_accepted = (
    df.groupby('LoanID')['Accepted']
    .apply(lambda x: 'true' in x.values)
    .reset_index(name='offer_accepted')
)

# Combine offer counts with acceptance data
offer_accept_analysis = pd.merge(
    offer_counts, 
    offer_accepted, 
    on='LoanID', 
    how='left'
)

# Calculate acceptance statistics by number of offers
acceptance_by_num_offers = (
    offer_accept_analysis
    .groupby('num_offers')
    .agg(
        total_cases=('LoanID', 'count'), 
        accepted_cases=('offer_accepted', 'sum')
    )
    .reset_index()
)

# Calculate acceptance rate as percentage
acceptance_by_num_offers['acceptance_rate'] = (
    acceptance_by_num_offers['accepted_cases'] / 
    acceptance_by_num_offers['total_cases'] * 100
)

print("Acceptance rate by number of offers per case:")
print(acceptance_by_num_offers.to_string(index=False, float_format='%.2f'))