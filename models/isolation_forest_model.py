import joblib
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class IsolationForestModel:
    def __init__(self, contamination=0.01, n_estimators=100, max_samples='auto', random_state=42):
        self.model = IsolationForest(contamination=contamination, n_estimators=n_estimators, max_samples=max_samples, random_state=random_state)
        self.model_path = 'isolation_forest_model.pkl'
    
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
        print(f"Training IsolationForest for {imeis_str} ...")

        # Feature selection and preprocessing
        features = ['sound_db', 'noise_db', 'breath_rate', 'heart_rate', 'temperature', 'humedity', 'time_in_minutes']
        X = df[features]

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Initialize and train the Isolation Forest model
        isolation_forest = IsolationForestModel(contamination=0.01, n_estimators=100, max_samples='auto', random_state=42)
        isolation_forest.train(X_scaled, model_path=f'isolation_forest_model_{imeis_str}.pkl')

        # Load the model for prediction
        isolation_forest.load_model(f'isolation_forest_model_{imeis_str}.pkl')

        # Predict and evaluate anomalies
        y_pred, anomalies = isolation_forest.predict(X_scaled)
        num_anomalies = isolation_forest.evaluate_anomalies(X_scaled)
        print(f"Predictions: {y_pred}")
        print(f"Anomalies: {anomalies}")
