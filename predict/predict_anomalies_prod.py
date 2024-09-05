import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from collections import Counter
import json

# Add the airo_ml directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import the models
from models.lstm_autoencoder_model import LSTMAutoencoderModel

from algorithms.anomaly_detectors import CustomAnomalyDetector
from iot.models import Message

# Check TensorFlow version and GPU availability
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Physical Devices:", tf.config.list_physical_devices())

PERCENTILE = 95
MIN_ANOMALY_COUNT = 5

class AnomalyPredictor:
    def __init__(self, imei, rows, model_file_path, batch_size=32, sequence_length=20):
        self.imei = imei
        self.rows = rows
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_file_path = model_file_path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.X_scaled, self.df, self.scaler = self.preprocess_data(rows)

    @staticmethod
    def preprocess_data(rows):
        df = pd.DataFrame(rows, columns=['sound_db', 'noise_db', 'breath_rate', 'heart_rate', 'temperature', 'humidity', 'dt_cr'])
        df['dt_cr'] = df['dt_cr'].apply(lambda x: x.hour * 60 + x.minute if isinstance(x, pd.Timestamp) else x)
        features = ['sound_db', 'noise_db', 'breath_rate', 'heart_rate', 'temperature', 'humidity', 'dt_cr']
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, df, scaler

    def create_sequences(self, data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequence = data[i:i + sequence_length]
            sequences.append(sequence)
        return np.array(sequences)

    def get_true_anomalous(self, anomaly_indices):
        if len(anomaly_indices) < 1:
            return []
        detector = CustomAnomalyDetector(neighbor_threshold=2, group_size_threshold=MIN_ANOMALY_COUNT)
        true_anomalies, outliers = detector.detect_anomalous_group(anomaly_indices)
        return true_anomalies

    def get_columns_with_highest_errors(self, reconstruction_errors, anomaly_indices, top_n=3):
        top_columns_counts = {f"top_{i+1}": Counter() for i in range(top_n)}

        for index in anomaly_indices:
            # Aggregate errors across time steps for the given sample
            error = np.mean(reconstruction_errors[index], axis=0)

            # Debugging statements
            if error.size != len(self.df.columns):
                print("Error: The size of the error array does not match the number of columns in the DataFrame.")
                continue

            # Get the column names for the top_n highest errors
            abnormal_columns = self.df.columns[np.argsort(error)[::-1]]
            print(f"Row {index} : {abnormal_columns[:3]}")  # Print top n columns with highest errors
            for i in range(top_n):
                top_columns_counts[f"top_{i+1}"].update([abnormal_columns[i]])


        return {k: dict(v) for k, v in top_columns_counts.items()}

    @staticmethod
    def aggregate_error_columns(aggregate_counter, new_counter):
        aggregate_counter.update(new_counter)
        return aggregate_counter

    @staticmethod
    def convert_aggregate_to_json(aggregate_counter):
        top_error_columns_dict = dict(aggregate_counter)
        top_error_columns_json = json.dumps(top_error_columns_dict)
        return top_error_columns_json

    def load_and_predict(self):
        X_sequences = self.create_sequences(self.X_scaled, self.sequence_length)
        X_time_series = np.reshape(X_sequences, (X_sequences.shape[0], self.sequence_length, X_sequences.shape[2]))

        # Load the LSTMAutoencoderModel model for prediction
        input_shape = (self.sequence_length, X_time_series.shape[2])
        lstm_autoencoder_loaded = LSTMAutoencoderModel(input_shape=input_shape)
        lstm_autoencoder_loaded.load_model(self.model_file_path)
        
        # Prediction
        anomalies_lstm, reconstruction_errors_lstm, mse_lstm = lstm_autoencoder_loaded.predict(X_time_series, percentile=PERCENTILE)

        anomaly_indices_lstm = np.where(anomalies_lstm)[0]
        
        true_lstm_indices = self.get_true_anomalous(anomaly_indices_lstm) if len(anomaly_indices_lstm) > 0 else []
        
        response_data = {
            "imei": self.imei,
            "true_lstm_indices": true_lstm_indices,
        }

        if len(true_lstm_indices) > 0:
            aggregate_counter = Counter()

            error_column_counter1 = self.get_columns_with_highest_errors(reconstruction_errors_lstm, true_lstm_indices)
            aggregate_counter = self.aggregate_error_columns(aggregate_counter, error_column_counter1)

            top_error_columns_json = self.convert_aggregate_to_json(aggregate_counter)
            response_data["top_error_columns"] = top_error_columns_json

        else:
            response_data["message"] = "not anomalous"

        return response_data

# Call the function to load models and predict anomalies
def load_models_and_predict_anomalies(imei, rows, model_file_path, batch_size=32, sequence_length=20):
    predictor = AnomalyPredictor(imei, rows, model_file_path, batch_size)
    return predictor.load_and_predict()

def fetch_rows_from_database(imei, limit):
    messages = Message.get(imei, limit)
    extracted_rows = [
        (msg.sound_db, msg.noise_db, msg.breath_rate, msg.heart_rate, msg.temperature, msg.humidity, msg.dt_cr)
        for msg in messages
    ]
    return extracted_rows

def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection using LSTM Autoencoder')
    parser.add_argument('--imei', required=True, help='IMEI of the device to fetch data')
    parser.add_argument('--limit', type=int, default=20, required=True, help='Number of messages to fetch from the database')
    parser.add_argument('--model_file_path', help='Path to the model file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction')
    parser.add_argument('--sequence_length', type=int, default=20, help='Sequence length for prediction')
    
    args = parser.parse_args()
    
    rows = fetch_rows_from_database(args.imei, args.limit)
    result = load_models_and_predict_anomalies(
        imei=args.imei,
        rows=rows,
        model_file_path=args.model_file_path,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length
    )
    
    print(result)

if __name__ == '__main__':
    main()