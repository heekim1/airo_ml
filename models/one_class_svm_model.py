import joblib
from sklearn.svm import OneClassSVM
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class OneClassSVMModel:
    def __init__(self, nu=0.01, kernel='rbf', gamma='scale'):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.model_path = 'one_class_svm_model.pkl'
    
    def train(self, X, model_path=None):
        self.model.fit(X)
        if model_path:
            self.model_path = model_path
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self, file_path=None):
        if file_path:
            self.model_path = file_path
        self.model = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")
    
    def predict(self, X):
        y_pred = self.model.predict(X)
        anomalies = (y_pred == -1)
        return y_pred, anomalies
    
    def evaluate_anomalies(self, X):
        _, anomalies = self.predict(X)
        num_anomalies = sum(anomalies)
        print(f"Number of anomalies detected: {num_anomalies}")
        return num_anomalies

# Example usage
if __name__ == "__main__":
    devices = [
        '883543430497535',
        #'701225054386494',
        #'683921756065543',
        #'605338565118998',
        #'486186400836117',
        #'463819615518786',
        #'460671778886265',
        #'353365188064688',
        #'159556169560848',
        #'148526548115987',
        #'117425803428623',
        #'108691961563855',
    ]
    batch_size = 32

    for device in devices:
        # Load data from CSV
        df = pd.read_csv(f'/Users/hkim75/Airo/airo_ml/utils/{device}_data_july_to_latest.csv')

        # Assuming the DataFrame has a column 'imei' to identify rows by device
        imeis = df['imei'].unique()
        # Convert to string
        imeis_str = ', '.join(map(str, imeis))
        print(f"Training OneClassSVM for {imeis_str} ...")

        # Feature selection and preprocessing
        features = ['sound_db', 'noise_db', 'breath_rate', 'heart_rate', 'temperature', 'humedity', 'time_in_minutes']
        X = df[features]

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Initialize and train the One-Class SVM model
        one_class_svm = OneClassSVMModel(nu=0.01, kernel='rbf', gamma='scale')
        one_class_svm.train(X_scaled, model_path=f'one_class_svm_model_{imeis_str}.pkl')

        # Load the model for prediction
        one_class_svm.load_model(f'one_class_svm_model_{imeis_str}.pkl')

        # Predict and evaluate anomalies
        y_pred, anomalies = one_class_svm.predict(X_scaled)
        num_anomalies = one_class_svm.evaluate_anomalies(X_scaled)
        print(f"Predictions: {y_pred}")
        print(f"Anomalies: {anomalies}")
