import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Download latest version
path = kagglehub.dataset_download("mbogernetto/brazilian-amazon-rainforest-degradation")

print("Path to dataset files:", path)

# Define dataset path
data_dir = path
def_area_file = os.path.join(data_dir, 'def_area_2004_2019.csv')

# Read the CSV file
def_area_df = pd.read_csv(def_area_file)

# Ensure 'Ano/Estados' column is treated as strings
def_area_df['Ano/Estados'] = def_area_df['Ano/Estados'].astype(str)

# Display basic information about the dataset
print("Deforestation Area Data:")
print(def_area_df.info())
print(def_area_df.head())

# Describe the dataset to get statistical insights
print(def_area_df.describe())

# Handle missing values, if any
def_area_df = def_area_df.dropna()

# Convert 'Ano/Estados' to 'Year'
def_area_df['Year'] = def_area_df['Ano/Estados']

# Shift the target column (AMZ LEGAL) up by one row to align with the previous year's data
def_area_df['AMZ LEGAL'] = def_area_df['AMZ LEGAL'].shift(-1)

# Drop the last row with NaN target value after shifting
def_area_df = def_area_df.dropna()

# Encode 'Year' as a categorical feature
def_area_df = pd.get_dummies(def_area_df, columns=['Year'], drop_first=True)

# Convert boolean columns to integers (0s and 1s)
for column in def_area_df.columns:
    if def_area_df[column].dtype == 'bool':
        def_area_df[column] = def_area_df[column].astype(int)

# Define features and target
features = def_area_df.drop(columns=['Ano/Estados', 'AMZ LEGAL'])  # Exclude original year column and total column
target = def_area_df['AMZ LEGAL']  # Using total deforestation area as the target for the next year

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Save the preprocessed data
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)














