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

# Check TensorFlow version and GPU availability
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Physical Devices:", tf.config.list_physical_devices())

# Function to check if GPU is being used
def is_gpu_available():
    return len(tf.config.list_physical_devices('GPU')) > 0

if is_gpu_available():
    print("GPU is available and will be used for training.")
else:
    print("GPU is not available. Training will use CPU.")


def train_model(imeis_str, batch_size, X, X_scaled, output="model"):

    # Train Isolation Forest
    print("Train Isolation Forest")
    isolation_forest = IsolationForestModel(contamination=0.01)
    isolation_forest.train(X_scaled, model_path=f'{output}/isolation_forest_model_{imeis_str}_{batch_size}.pkl')
    """
    # Train One-Class SVM
    print("Train One-Class SVM")
    one_class_svm = OneClassSVMModel(nu=0.01, kernel="rbf", gamma=0.1)
    one_class_svm.train(X_scaled, model_path=f'{output}/one_class_svm_model_{imeis_str}_{batch_size}.pkl')
    """

    # Train Autoencoder
    print("Train Autoencoder")
    input_dim = X_scaled.shape[1]
    autoencoder = AutoencoderModel(input_dim=input_dim)
    history = autoencoder.train(X_scaled, batch_size, epochs=12)
    autoencoder.save_model(f'{output}/autoencoder_model_{imeis_str}_{batch_size}.keras')
    print(f"Autoencoder - Batch Size: {batch_size}, Loss: {history.history['loss'][-1]}, Val Loss: {history.history['val_loss'][-1]}")

    # Train LSTM-based model for time-series data
    print("Train LSTM-based model for time-series data")
    # Reshape data for LSTM model
    X_time_series = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
    input_shape = (X_time_series.shape[1], X_time_series.shape[2])

    # Initialize and train the LSTM autoencoder model
    lstm_autoencoder = LSTMAutoencoderModel(input_shape=input_shape)
    history = lstm_autoencoder.train(X_time_series, batch_size, epochs=12)
    lstm_autoencoder.save_model(f'{output}/lstm_autoencoder_model_{imeis_str}_{batch_size}.keras')

    print(f"LSTM Autoencoder - Batch Size: {batch_size}, Loss: {history.history['loss'][-1]}, Val Loss: {history.history['val_loss'][-1]}")
    

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

for device in devices:
    # Load data from CSV
    df = pd.read_csv(f'/Users/hkim75/Airo/airo_ml/utils/{device}_data_july_to_latest.csv')

    # Assuming the DataFrame has a column 'imei' to identify rows by device
    imeis = df['imei'].unique()
    # Convert to string
    imeis_str = ', '.join(map(str, imeis))
    print(f"Training models for {imeis_str} ...")

    # Feature selection and preprocessing
    features = ['sound_db', 'noise_db', 'breath_rate', 'heart_rate', 'temperature', 'humedity', 'time_in_minutes']
    X = df[features]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test different batch sizes
    for batch_size in [32]: #[32, 64, 128, 256]:
        print(f"Testing batch size: {batch_size}")
        train_model(imeis_str, batch_size, X, X_scaled, output="/Users/hkim75/Airo/airo_ml/train/model_3_single_07132024")
