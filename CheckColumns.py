import pandas as pd

# Load the preprocessed data (or new data for prediction)
new_data = pd.read_csv('X_val.csv')  # Replace with the path to your new data CSV file

# Display the columns in new_data to verify
print("Columns in new_data:", new_data.columns)
