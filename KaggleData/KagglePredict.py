import torch
import pandas as pd
from KaggleModelClass import SimpleNN

# Define the function to predict deforestation for the year 2020
def predict_deforestation_2020(model_path, feature_data_path, device):
    # Load the preprocessed features
    features_df = pd.read_csv(feature_data_path)

    # Ensure the data for the last available year (e.g., 2019) is selected
    # This assumes that the last row in features_df corresponds to the year 2019
    features_2019 = features_df.iloc[-1:]  # Keep all features, including Year_ columns

    # Convert the feature data to a torch tensor
    features_2019_tensor = torch.tensor(features_2019.values, dtype=torch.float32).to(device)
    
    # Load the trained model
    input_size = features_2019.shape[1]  # Number of features
    model = SimpleNN(input_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # Set the model to evaluation mode

    # Perform prediction
    with torch.no_grad():  # Disable gradients
        prediction = model(features_2019_tensor)
    
    return prediction.item()

# Define file paths and device
model_path = 'deforestation_prediction_model.pth'
feature_data_path = 'X_train.csv'  # Use the training data file for demonstration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Predict deforestation for 2020
deforestation_2020 = predict_deforestation_2020(model_path, feature_data_path, device)
print(f"Predicted Total Deforestation for 2020: {deforestation_2020}")