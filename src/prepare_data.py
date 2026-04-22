import pandas as pd
import os

# Load both files
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

# Add labels: FAKE = 1, TRUE = 0
fake_df['label'] = 1
true_df['label'] = 0

# Combine and shuffle
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df[['title', 'text', 'label']]
df.dropna(inplace=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save cleaned dataset
os.makedirs('data', exist_ok=True)
df.to_csv('data/train.csv', index=False)

print("✅ Saved cleaned dataset to data/train.csv")
