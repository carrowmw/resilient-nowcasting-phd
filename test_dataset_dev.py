import pandas as pd
import numpy as np

# Read small subset to see columns exactly
df = pd.read_parquet('data/processed/master_dataset_15T.parquet', filters=[('dt', '>=', pd.Timestamp('2023-07-01')), ('dt', '<', pd.Timestamp('2023-07-02'))])
print(df.columns)
print(df.head())
print("Unique locations:", len(df['location'].unique()))
print("Unique times:", len(df['dt'].unique()))
