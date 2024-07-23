# Function to train and evaluate the model
import os,sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from collections import Counter
import json

from database_service import get_mysql_connection, get_latest_20_data, get_latest_60_data, get_latest_120_data, get_latest_180_data, get_latest_240_data

# Add the airo_ml directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import the models
from models.autoencoder_model import AutoencoderModel
from models.isolation_forest_model import IsolationForestModel
from models.one_class_svm_model import OneClassSVMModel
from models.lstm_autoencoder_model import LSTMAutoencoderModel

from algorithms.anomaly_detectors import CustomAnomalyDetector


# Check TensorFlow version and GPU availability
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Physical Devices:", tf.config.list_physical_devices())


PERCENTILE=95
MIN_ANOMALY_COUNT=5


# Preprocess the data
def preprocess_data(rows):
    df = pd.DataFrame(rows, columns=['sound_db', 'noise_db', 'breath_rate', 'heart_rate', 'temperature', 'humidity', 'dt_cr'])
    df['dt_cr'] = df['dt_cr'].apply(lambda x: x.hour * 60 + x.minute if isinstance(x, pd.Timestamp) else x)
    #df = df[['sound_db', 'noise_db', 'breath_rate', 'heart_rate', 'temperature', 'humidity', 'time_in_minutes']]
    features = ['sound_db', 'noise_db', 'breath_rate', 'heart_rate', 'temperature', 'humidity', 'dt_cr']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, df

def get_sample_data():
    # Load the dataset
    df = pd.read_csv('/Users/heeshinkim/Desktop/Airosolution/ml/data/sample_dataset.csv')

    # Convert the 'dt_cr' column to datetime if necessary
    df['dt_cr'] = pd.to_datetime(df['dt_cr'], unit='m')

    # Preprocess the data
    #X_scaled, df_processed = preprocess_data(df.values)

    print(df.head)

    return df

def create_anomalous_dataset():
    # Define the number of minutes in a day
    num_minutes = 100
    
    # Create a sample dataset with specific anomalies for testing
    data = {
        'sound_db': np.random.uniform(30, 100, num_minutes),
        'noise_db': np.random.uniform(20, 90, num_minutes),
        'breath_rate': np.random.uniform(12, 20, num_minutes),
        'heart_rate': np.random.uniform(60, 100, num_minutes),
        'temperature': np.random.uniform(36, 37.5, num_minutes),
        'humidity': np.random.uniform(30, 70, num_minutes),
        'dt_cr': pd.date_range(start='2023-07-15 00:00:00', periods=num_minutes, freq='T')
    }

    df = pd.DataFrame(data)

    # Introduce anomalies between 11:00 PM and 5:00 AM
    anomalous_hours = (df['dt_cr'].dt.hour >= 23) | (df['dt_cr'].dt.hour < 5)
    df.loc[anomalous_hours, 'breath_rate'] = np.random.uniform(6, 10, anomalous_hours.sum())
    df.loc[anomalous_hours, 'heart_rate'] = np.random.uniform(40, 55, anomalous_hours.sum())

    # Convert 'dt_cr' to minutes since midnight
    df['dt_cr_minutes'] = df['dt_cr'].dt.hour * 60 + df['dt_cr'].dt.minute
    return df


def get_true_anomalous(anomaly_indices):
    print(f">>>>>get_true_anomalus : {len(anomaly_indices)}")
    if len(anomaly_indices) < 1:
        return []
    
    detector = CustomAnomalyDetector(neighbor_threshold=2, group_size_threshold=MIN_ANOMALY_COUNT)
    true_anomalies, outliers = detector.detect_anomalous_group(anomaly_indices)

    return true_anomalies

def get_columns_with_highest_errors(reconstruction_errors_lstm, anomaly_indices, df, top_n=3):
    column_error_counts = Counter()

    for index in anomaly_indices:
        error = reconstruction_errors_lstm[index].flatten()
        abnormal_columns = df.columns[np.argsort(error)[::-1]]
        column_error_counts.update(abnormal_columns[:top_n])
        print(f"Row {index}: {abnormal_columns[:top_n]}")  # Print top n columns with highest errors

    return column_error_counts

# Function to aggregate results from multiple calls
def aggregate_error_columns(aggregate_counter, new_counter):
    aggregate_counter.update(new_counter)
    return aggregate_counter

# Convert the final aggregated results to JSON
def convert_aggregate_to_json(aggregate_counter):
    top_error_columns_dict = dict(aggregate_counter)
    top_error_columns_json = json.dumps(top_error_columns_dict)
    return top_error_columns_json

# Load models and predict anomalies for each device
def load_models_and_predict_anomalies():
    #devices = airo_device.get_all()
    devices = [
        '883543430497535',
        '701225054386494',
        '683921756065543',
        '605338565118998',
        '486186400836117',
        '463819615518786',
        '460671778886265',
        '353365188064688',
        '159556169560848',
        '148526548115987',
        '117425803428623',
        '108691961563855',
    ]
    batch_size=32
    input_dir = "/Users/heeshinkim/Desktop/Airosolution/ml/train/model_3_single_07132024"

    anomalous_df = create_anomalous_dataset()

    for device in devices:
        print(f"======================================= Device : {device} =======================================")
        
        rows = get_latest_240_data(device)
        if len(rows) < 240:
            print(f"Not enough data for device {device}")
            continue

        X_scaled, df = preprocess_data(rows)
        
        #X_scaled, df = preprocess_data(anomalous_df)
        X_time_series = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))


        # Load the LSTMAutoencoderModel model for prediction
        input_shape = (X_time_series.shape[1], X_time_series.shape[2])
        lstm_autoencoder_loaded = LSTMAutoencoderModel(input_shape=input_shape)
        lstm_autoencoder_loaded.load_model(f'{input_dir}/lstm_autoencoder_model_{device}_{batch_size}.keras')
        # prediction
        mse_lstm = lstm_autoencoder_loaded.evaluate_loss(X_time_series)
        loss_cutoff_lstm = lstm_autoencoder_loaded.determine_loss_cutoff(mse_lstm, percentile=PERCENTILE)
        #print(f"LSTMAutoencoderModel Loss Cutoff: {loss_cutoff_lstm}")
        anomalies_lstm, mse_lstm = lstm_autoencoder_loaded.predict(X_time_series, loss_cutoff_lstm)
        #print(f"LSTMAutoencoderModel Anomalies Detected: {anomalies_lstm}")


        # Load the LSTMAutoencoderModel model for all devices
        all_lstm_autoencoder_loaded = LSTMAutoencoderModel(input_shape=input_shape)
        all_lstm_autoencoder_loaded.load_model(f'{input_dir}/lstm_autoencoder_model_all_devices.keras')
        # prediction
        mse_all_lstm = all_lstm_autoencoder_loaded.evaluate_loss(X_time_series)
        loss_cutoff_all_lstm = all_lstm_autoencoder_loaded.determine_loss_cutoff(mse_all_lstm, percentile=PERCENTILE)
        #print(f"LSTMAutoencoderModel Loss Cutoff: {loss_cutoff_all_lstm}")
        anomalies_all_lstm, mse_all_lstm = all_lstm_autoencoder_loaded.predict(X_time_series, loss_cutoff_all_lstm)
        #print(f"LSTMAutoencoderModel Anomalies Detected: {anomalies_all_lstm}")

        
        anomaly_indices_lstm = np.where(anomalies_lstm)[0]
        anomaly_indices_all_lstm = np.where(anomalies_all_lstm)[0]
        
        true_lstm_indices = get_true_anomalous(anomaly_indices_lstm) if len(anomaly_indices_lstm) > 0 else []
        true_all_lstm_indices = get_true_anomalous(anomaly_indices_all_lstm) if len(anomalies_all_lstm) > 0 else []
        print(f"true_lstm_indices: {true_lstm_indices}")
        print(f"true_all_lstm_indices: {true_all_lstm_indices}")

        if len(true_lstm_indices) > 0 and len(true_all_lstm_indices) > 0:
            print(f"Device {device} -  detected anomalies")
            # Print the rows with anomalies for each model
            print(f"Device {device} - Anomalous Rows (LSTM Autoencoder):")
            print(df.iloc[true_lstm_indices])
            print(f"Device {device} - Anomalous Rows (Global LSTM Autoencoder):")
            print(df.iloc[true_all_lstm_indices])

            # Example usage
            aggregate_counter = Counter()

            # Get reconstruction errors for autoencoder models
            reconstruction_errors_lstm = lstm_autoencoder_loaded.get_reconstruction_errors(X_time_series)
            reconstruction_errors_all_lstm = all_lstm_autoencoder_loaded.get_reconstruction_errors(X_time_series)

            print(f"Device {device} - Columns with highest reconstruction errors (LSTM Autoencoder):")
            error_column_counter1 = get_columns_with_highest_errors(reconstruction_errors_lstm, true_lstm_indices, df)
            aggregate_counter = aggregate_error_columns(aggregate_counter, error_column_counter1)

            print(f"Device {device} - Columns with highest reconstruction errors (Global LSTM Autoencoder):")
            error_column_counter2 = get_columns_with_highest_errors(reconstruction_errors_all_lstm, true_all_lstm_indices, df)
            aggregate_counter = aggregate_error_columns(aggregate_counter, error_column_counter2)

            top_error_columns_json = convert_aggregate_to_json(aggregate_counter)
            print(top_error_columns_json)

        elif len(true_lstm_indices) > 0 or len(true_all_lstm_indices) > 0:
            print(f"Device {device} -  warning")
        else:
            print(f"Device {device} -  not anomalous")





# Call the function to load models and predict anomalies
load_models_and_predict_anomalies()
