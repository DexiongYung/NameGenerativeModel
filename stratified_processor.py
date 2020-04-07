"""
1. Append the length column to all names.
2. Append P(name|length) column to all probs: Calculated by count(name) / count(names of length).

3. During training, provide P(length) for all lengths.
4. P(name, length) is then computed by P(name|length) * P(length).

5. Assign row weights as P(name, length). Finished!
"""

import pandas as pd

CSV_PATH = 'Data/LastNames.csv'

df = pd.read_csv(CSV_PATH)
df = df[:500000]
df = df[['name','count']]
df = df[df['name'].apply(lambda x: len(x) < 10)]
df['length'] = 0
df['p_given_length'] = 0.
length_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(len(df)):
    if i%1000 == 0: print(f"Loop 1 Iteration: {i}")
    length = len(df.iloc[i]['name'])
    df['length'] = length
    length_counts[length-1] += 1

for i in range(len(df)):
    if i%1000 == 0: print(f"Loop 2 Iteration: {i}")
    length = len(df.iloc[i]['name'])
    df['p_name_given_length'] = df.iloc[i]['count']/length_counts[length-1]

df.to_csv(CSV_PATH, index=False)