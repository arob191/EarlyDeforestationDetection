import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Download latest version
path = kagglehub.dataset_download("mbogernetto/brazilian-amazon-rainforest-degradation")

print("Path to dataset files:", path)

# Define dataset path
data_dir = path  # Corrected to avoid double concatenation
def_area_file = os.path.join(data_dir, 'def_area_2004_2019.csv')

# Check if the file exists
if not os.path.exists(def_area_file):
    print(f"File not found: {def_area_file}")
else:
    print(f"File found: {def_area_file}")

# Read the CSV file
def_area_df = pd.read_csv(def_area_file)

# Display basic information about the dataset
print("Deforestation Area Data:")
print(def_area_df.info())
print(def_area_df.head())

# Describe the dataset to get statistical insights
print(def_area_df.describe())

# Handle missing values, if any
def_area_df = def_area_df.dropna()

# Define features and target
features = def_area_df.drop(columns=['Ano/Estados', 'AMZ LEGAL'])  # Exclude year and total column for individual predictions
target = def_area_df['AMZ LEGAL']  # Using total deforestation area as the target for demonstration

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Save the preprocessed data
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)











