import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load CSV
df = pd.read_csv('call_metadata.csv')
print("Original data sample:")
print(df.head())

# Check missing values and remove duplicates
print("Missing values:\n", df.isnull().sum())
df = df.drop_duplicates()

# Encode categorical features
le_location = LabelEncoder()
df['location'] = le_location.fit_transform(df['location'])

le_calltype = LabelEncoder()
df['call_type'] = le_calltype.fit_transform(df['call_type'])

# Normalize numeric features
scaler = MinMaxScaler()
df[['duration','frequency']] = scaler.fit_transform(df[['duration','frequency']])

print("Processed metadata sample:")
print(df.head())

