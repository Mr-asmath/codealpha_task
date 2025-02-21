import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')  # Set the backend to a non-interactive one (agg)

import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Define Neural Network Model
class MultiDatasetModel(nn.Module):
    def __init__(self, input_size):
        super(MultiDatasetModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

# Load datasets (for illustration purposes; not used in prediction)
df1 = pd.read_csv('datasets/heart.csv')
df2 = pd.read_csv('datasets/heart_1.csv')
df3 = pd.read_csv('datasets/heart_2.csv')
datasets = [df1, df2, df3]

# Load models and scalers (previously trained and saved)
models = {}
scalers = {}

# Function to load models and scalers
def load_models_and_scalers():
    # Load Neural Network model
    input_size = 13  # Assuming this is the input size of your neural network
    model_nn = MultiDatasetModel(input_size)
    model_nn.load_state_dict(torch.load('models/model_nn.pth'))
    model_nn.eval()
    models['Neural Network'] = model_nn

    # Load Logistic Regression and Random Forest models
    algorithms = ['Logistic Regression', 'Random Forest']
    for alg_name in algorithms:
        for dataset_idx in range(1, 4):  # Assuming datasets are numbered from 1 to 3
            model_path = f'models/{alg_name}_dataset{dataset_idx}.pkl'
            scaler_path = f'models/{alg_name}_dataset{dataset_idx}_scaler.pkl'

            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                models[f'{alg_name}_dataset{dataset_idx}'] = model
                scalers[f'{alg_name}_dataset{dataset_idx}'] = scaler
            except FileNotFoundError:
                print(f"Error: Model file {model_path} or scaler file {scaler_path} not found.")
                continue

# Call the function to load models and scalers
load_models_and_scalers()

# Flask route to handle home page
@app.route('/')
def home():
    return render_template('index.html')

# Flask route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Preprocess input data
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Predictions dictionary
        predictions = {}

        # Make prediction with Neural Network
        model_nn = models['Neural Network']
        scaler_nn = StandardScaler()  # Assuming the same scaler for NN
        input_scaled_nn = scaler_nn.fit_transform(input_data)
        input_tensor_nn = torch.tensor(input_scaled_nn, dtype=torch.float32)
        with torch.no_grad():
            output_nn = model_nn(input_tensor_nn).numpy()
            prediction_nn = (output_nn > 0.5).astype(int)[0][0]
        predictions['Neural Network'] = prediction_nn

        # Load and predict with Logistic Regression and Random Forest models
        for alg_name in ['Logistic Regression', 'Random Forest']:
            for dataset_idx in range(1, 4):  # Assuming datasets are numbered from 1 to 3
                model = models[f'{alg_name}_dataset{dataset_idx}']
                scaler = scalers[f'{alg_name}_dataset{dataset_idx}']
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                predictions[f'{alg_name}_dataset{dataset_idx}'] = prediction

        # Determine prediction message
        if prediction_nn == 1:
            result = 'High Risk of Heart Disease'
        else:
            result = 'Low Risk of Heart Disease'

        # Generate a bar chart for predictions
        plt.figure(figsize=(10, 6))
        algorithms = ['Neural Network', 'Logistic Regression_dataset1', 'Logistic Regression_dataset2',
                      'Logistic Regression_dataset3', 'Random Forest_dataset1', 'Random Forest_dataset2',
                      'Random Forest_dataset3']
        values = [predictions[alg] for alg in algorithms]
        plt.bar(algorithms, values, color=['blue', 'green', 'green', 'green', 'orange', 'orange', 'orange'])
        plt.title('Heart Disease Risk Prediction')
        plt.xlabel('Algorithms')
        plt.ylabel('Risk Level (0 or 1)')
        plt.ylim(0, 1)
        plt.grid(True)

        # Convert plot to base64 for embedding in HTML
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        graph_url = base64.b64encode(buffer.getvalue()).decode()
        plt.close()  # Clear the plot to release memory

        return render_template('result.html', prediction_result=result, predictions=predictions, graph_url=graph_url)

if __name__ == '__main__':
    app.run(debug=True)
