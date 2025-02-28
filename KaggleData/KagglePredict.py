import torch
import pandas as pd
from KaggleModelClass import SimpleNN

# Load the preprocessed data (or new data for prediction)
new_data = pd.read_csv('X_val.csv')  # Replace with the path to your new data CSV file

# Create a list of years (assuming 2019 for this example)
years = [2019] * new_data.shape[0]
states = new_data.columns.tolist()

# Flatten the new_data DataFrame
flat_data = new_data.melt(var_name='State', value_name='Deforestation_Area')

# Assuming same year for all data, create a column for years
flat_data['Year'] = [2019] * flat_data.shape[0]

# Display the flattened data to verify
print("Flattened data:")
print(flat_data.head())

# Prepare the input data for prediction
areas = flat_data['State']
inputs = pd.get_dummies(flat_data[['State', 'Year']], drop_first=True)  # Encode categorical data

# Ensure all input features are of numeric types
inputs = inputs.apply(pd.to_numeric, errors='coerce').fillna(0)

# Convert boolean columns to integers
inputs = inputs.astype(float)

# Verify the input data shape
print("Shape of inputs for prediction:", inputs.shape)
print("Inputs for prediction:")
print(inputs.head())

# Check if inputs DataFrame is empty
if inputs.empty:
    raise ValueError("Input data is empty. Ensure the input data is correctly formatted.")

# Convert data to torch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = inputs.shape[1]
model = SimpleNN(input_size).to(device)

# Load the trained model
model.load_state_dict(torch.load('deforestation_prediction_model.pth', map_location=device))
model.eval()

# Convert input data to torch tensors
inputs_tensor = torch.tensor(inputs.values, dtype=torch.float32).to(device)

# Make predictions
with torch.no_grad():
    predictions = model(inputs_tensor)

# Convert predictions to DataFrame and add state and year columns
predictions_df = pd.DataFrame(predictions.cpu().numpy(), columns=['Predicted_Deforestation_Area'])
predictions_df['State'] = areas.values
predictions_df['Year'] = flat_data['Year'].values

# Save the predictions to CSV
predictions_df.to_csv('predictions_with_info.csv', index=False)

print("Predictions saved to predictions_with_info.csv")

