# Function to train and evaluate the model
import os,sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Add the airo_ml directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import the models
from models.autoencoder_model import AutoencoderModel
from models.isolation_forest_model import IsolationForestModel
from models.one_class_svm_model import OneClassSVMModel
from models.lstm_autoencoder_model import LSTMAutoencoderModel


print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Physical Devices:", tf.config.list_physical_devices())

# Function to check if GPU is being used
def is_gpu_available():
    from tensorflow.python.client import device_lib
    devices = device_lib.list_local_devices()
    gpu_devices = [device for device in devices if device.device_type == 'GPU']
    return len(gpu_devices) > 0

if is_gpu_available():
    print("GPU is available and will be used for training.")
else:
    print("GPU is not available. Training will use CPU.")

output = "model_3_single_07132024"

batch_size = 64

# Load data from CSV
df = pd.read_csv('/Users/heeshinkim/Desktop/Airosolution/ml/data/imei_data_btw_july_1st_july_15_2024.csv')

# Feature selection and preprocessing
features = ['sound_db', 'noise_db', 'breath_rate', 'heart_rate', 'temperature', 'humedity', 'time_in_minutes']
X = df[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""
# Train Isolation Forest
print("Train Isolation Forest")
isolation_forest = IsolationForestModel(contamination=0.01)
isolation_forest.train(X_scaled, model_path=f'{output}/isolation_forest_model_all_devices.pkl')


# Train One-Class SVM
print("Train One-Class SVM")
one_class_svm = OneClassSVMModel(nu=0.01, kernel="rbf", gamma=0.1)
one_class_svm.train(X_scaled, model_path=f'{output}/one_class_svm_model_all_devices.pkl')
"""

# Train Autoencoder
print("Train Autoencoder")
input_dim = X_scaled.shape[1]
autoencoder = AutoencoderModel(input_dim=input_dim)
history = autoencoder.train(X_scaled, batch_size, epochs=12)
autoencoder.save_model(f'{output}/autoencoder_model_all_devices.keras')
print(f"Autoencoder - Batch Size: {batch_size}, Loss: {history.history['loss'][-1]}, Val Loss: {history.history['val_loss'][-1]}")

# Train LSTM-based model for time-series data
print("Train LSTM-based model for time-series data")
# Reshape data for LSTM model
X_time_series = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
input_shape = (X_time_series.shape[1], X_time_series.shape[2])

# Initialize and train the LSTM autoencoder model
lstm_autoencoder = LSTMAutoencoderModel(input_shape=input_shape)
history = lstm_autoencoder.train(X_time_series, batch_size, epochs=12)
lstm_autoencoder.save_model(f'{output}/lstm_autoencoder_model_all_devices.keras')

print(f"LSTM Autoencoder - Batch Size: {batch_size}, Loss: {history.history['loss'][-1]}, Val Loss: {history.history['val_loss'][-1]}")
