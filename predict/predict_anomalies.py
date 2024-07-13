import MySQLdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import sys
import os
from sklearn.preprocessing import StandardScaler

# Add the models directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

from isolation_forest_model import IsolationForestModel  # Import the IsolationForestModel class
from one_class_svm_model import OneClassSVMModel  # Import the OneClassSVMModel class
from autoencoder_model import AutoencoderModel  # Import the AutoencoderModel class
from lstm_autoencoder_model import LSTMAutoencoderModel, determine_loss_cutoff  # Import the LSTMAutoencoderModel class

# Load environment variables from .env file
load_dotenv()

DATABASES = {
    "mysql": {
        "ENGINE": os.environ.get("DB_ENGINE"),
        "NAME": os.environ.get("SQL_NAME"),
        "USER": os.environ.get("SQL_USER"),
        "PASSWORD": os.environ.get("SQL_PASSWORD"),
        "HOST": os.environ.get("SQL_HOST"),
        "PORT": os.environ.get("SQL_PORT"),
    },
}
print(f"DATABASES: {DATABASES}")

def get_mysql_connection():
    db_settings = DATABASES["mysql"]
    connection = MySQLdb.connect(
        host=db_settings["HOST"],
        user=db_settings["USER"],
        password=db_settings["PASSWORD"],
        database=db_settings["NAME"],
        port=int(db_settings["PORT"]),
    )
    return connection

class airo_device:
    def __init__(self, imei, sim, type, status, dt_cr, dt_up):
        self.imei = imei
        self.sim = sim
        self.type = type
        self.status = status
        self.dt_cr = dt_cr
        self.dt_up = dt_up

    @staticmethod
    def get_all():
        connection = get_mysql_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT imei, sim, type, status, dt_cr, dt_up FROM airo_device")
        devices = [
            airo_device(imei, sim, type, status, dt_cr, dt_up)
            for imei, sim, type, status, dt_cr, dt_up in cursor.fetchall()
        ]
        connection.close()
        return devices

def get_latest_20_data(imei):
    connection = get_mysql_connection()
    cursor = connection.cursor()
    query = """
        SELECT sound_db, noise_db, breath_rate, heart_rate, temperature, humedity, dt_cr
        FROM airo_message
        WHERE imei = %s
        ORDER BY dt_cr DESC
        LIMIT 20
    """
    cursor.execute(query, (imei,))
    rows = cursor.fetchall()
    connection.close()
    return rows

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

def get_feature_contributions(model, X, method='mean'):
    if method == 'mean':
        if X.size == 0:
            return np.array([])
        mean = np.mean(X, axis=0)
        contributions = np.abs(X - mean)
    else:
        raise ValueError(f"Unknown method: {method}")
    return contributions


# Load models and predict anomalies for each device
def load_models_and_predict_anomalies():
    devices = airo_device.get_all()

    model_root = "/Users/heeshinkim/Desktop/Airosolution/ml/train"
    isolation_forest_model = IsolationForestModel()
    isolation_forest_model.load(file_path=f'{model_root}/isolation_forest_model.pkl')

    one_class_svm_model = OneClassSVMModel()
    one_class_svm_model.load(file_path=f'{model_root}/one_class_svm_model.pkl')

    autoencoder_model = AutoencoderModel(input_dim=7)
    autoencoder_model.load(file_path=f'{model_root}/autoencoder_model.h5')

    lstm_autoencoder_model = LSTMAutoencoderModel(input_shape=(20, 7))
    lstm_autoencoder_model.load(file_path=f'{model_root}/lstm_autoencoder_model.h5')


    for device in devices:
        rows = get_latest_20_data(device.imei)
        if len(rows) < 20:
            print(f"Not enough data for device {device.imei}")
            continue

        X_scaled, df = preprocess_data(rows)
        X_time_series = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

        # Predict anomalies
        anomalies_if = isolation_forest_model.predict(X_scaled)
        anomalies_svm = one_class_svm_model.predict(X_scaled)

        # Evaluate and print loss for Autoencoder
        mse_ae = autoencoder_model.evaluate_loss(X_scaled)
        threshold_ae = determine_loss_cutoff(mse_ae, 95)
        anomalies_ae = autoencoder_model.predict(X_scaled, loss_cutoff=threshold_ae)

        # Evaluate and print loss for LSTM Autoencoder
        mse_lstm = lstm_autoencoder_model.evaluate_loss(X_time_series)
        threshold_lstm = determine_loss_cutoff(mse_lstm, 95)
        anomalies_lstm = lstm_autoencoder_model.predict(X_time_series, loss_cutoff=threshold_lstm)

        print(f"Device {device.imei} - Isolation Forest detected {np.sum(anomalies_if)} anomalies")
        print(f"Device {device.imei} - One-Class SVM detected {np.sum(anomalies_svm)} anomalies")
        print(f"Device {device.imei} - Autoencoder detected {np.sum(anomalies_ae)} anomalies")
        print(f"Device {device.imei} - LSTM Autoencoder detected {np.sum(anomalies_lstm)} anomalies")

        # Print the rows with anomalies for each model
        anomaly_indices_if = np.where(anomalies_if)[0]
        anomaly_indices_svm = np.where(anomalies_svm)[0]
        anomaly_indices_ae = np.where(anomalies_ae)[0]
        anomaly_indices_lstm = np.where(anomalies_lstm)[0]

        print(f"Device {device.imei} - Anomalous Rows (Isolation Forest):")
        print(df.iloc[anomaly_indices_if])
        print(f"Device {device.imei} - Anomalous Rows (One-Class SVM):")
        print(df.iloc[anomaly_indices_svm])
        print(f"Device {device.imei} - Anomalous Rows (Autoencoder):")
        print(df.iloc[anomaly_indices_ae])
        print(f"Device {device.imei} - Anomalous Rows (LSTM Autoencoder):")
        print(df.iloc[anomaly_indices_lstm])

        # Get feature contributions for Isolation Forest and One-Class SVM
        contributions_if = get_feature_contributions(isolation_forest_model, X_scaled[anomaly_indices_if])
        contributions_svm = get_feature_contributions(one_class_svm_model, X_scaled[anomaly_indices_svm])

        print(f"Device {device.imei} - Columns with highest feature contributions (Isolation Forest):")
        for i, index in enumerate(anomaly_indices_if):
            contributions = contributions_if[i]
            abnormal_columns = df.columns[np.argsort(contributions)[::-1]]
            print(f"Row {index}: {abnormal_columns[:3]}")  # Print top 3 columns with highest contributions

        print(f"Device {device.imei} - Columns with highest feature contributions (One-Class SVM):")
        for i, index in enumerate(anomaly_indices_svm):
            contributions = contributions_svm[i]
            abnormal_columns = df.columns[np.argsort(contributions)[::-1]]
            print(f"Row {index}: {abnormal_columns[:3]}") 

        # Get reconstruction errors for autoencoder models
        reconstruction_errors_ae = autoencoder_model.get_reconstruction_errors(X_scaled)
        reconstruction_errors_lstm = lstm_autoencoder_model.get_reconstruction_errors(X_time_series)

        # Print the columns with the highest reconstruction errors
        print(f"Device {device.imei} - Columns with highest reconstruction errors (Autoencoder):")
        for index in anomaly_indices_ae:
            error = reconstruction_errors_ae[index]
            abnormal_columns = df.columns[np.argsort(error)[::-1]]
            print(f"Row {index}: {abnormal_columns[:3]}")  # Print top 3 columns with highest errors

        print(f"Device {device.imei} - Columns with highest reconstruction errors (LSTM Autoencoder):")
        for index in anomaly_indices_lstm:
            error = reconstruction_errors_lstm[index].flatten()
            abnormal_columns = df.columns[np.argsort(error)[::-1]]
            print(f"Row {index}: {abnormal_columns[:3]}")  # Print top 3 columns with highest errors

# Call the function to load models and predict anomalies
load_models_and_predict_anomalies()

