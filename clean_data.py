import pandas as pd

df = pd.read_csv('customer_data.csv')

# 1. Shape — how many rows and columns
print("Shape:", df.shape)

# 2. Missing values — any empty cells?
print("\nMissing values:")
print(df.isnull().sum())

# 3. Data types — is each column the right type?
print("\nData types:")
print(df.dtypes)

# 4. Duplicates — any repeated rows?
print(f"\nDuplicates: {df.duplicated().sum()}")

# 5. Basic statistics
print("\nBasic stats:")
print(df.describe())