import numpy as np
import pandas as pd

df = pd.read_csv('emails.csv')

print(df.isna().sum())
df = df.fillna(0)

print(df)

df['BooleanSpam'] = (df['Category'] == 'spam').astype(int)

print(df)