import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Dropout
from keras.optimizers import Adam
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2

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


# Function to train and evaluate the model
def train_model(imeis_str, batch_size, X, X_scaled):
    """
    print("Train Isolation Forest")
    isolation_forest = IsolationForest(contamination=0.01)
    isolation_forest.fit(X_scaled)
    joblib.dump(isolation_forest, f'isolation_forest_model_{imeis_str}_{batch_size}.pkl')


    print("Train One-Class SVM")
    one_class_svm = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1)
    one_class_svm.fit(X_scaled)
    joblib.dump(one_class_svm, f'one_class_svm_model_{imeis_str}_{batch_size}.pkl')


    print("Train Autoencoder")
    input_dim = X_scaled.shape[1]
    encoding_dim = 14
    hidden_dim = int(encoding_dim / 2)

    autoencoder = Sequential()
    autoencoder.add(Input(shape=(input_dim,)))
    autoencoder.add(Dense(encoding_dim, activation="relu"))
    autoencoder.add(Dense(hidden_dim, activation="relu"))
    autoencoder.add(Dense(encoding_dim, activation="relu"))
    autoencoder.add(Dense(input_dim, activation="sigmoid"))
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    history = autoencoder.fit(X_scaled, X_scaled, epochs=12, batch_size=batch_size, validation_split=0.1, verbose=1)
    autoencoder.save(f'autoencoder_model_{imeis_str}_{batch_size}.h5')
    print(f"Autoencoder - Batch Size: {batch_size}, Loss: {history.history['loss'][-1]}, Val Loss: {history.history['val_loss'][-1]}")
    """

    print("Train LSTM-based model for time-series data")
    # Reshape data for LSTM model
    X_time_series = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
    print("X_time_series shape:", X_time_series.shape)
    input_shape = (X_time_series.shape[1], X_time_series.shape[2])
    print("input_shape:", input_shape)

    lstm_autoencoder = Sequential()
    lstm_autoencoder.add(Input(shape=input_shape))
    lstm_autoencoder.add(LSTM(100, activation='relu', return_sequences=True))
    lstm_autoencoder.add(LSTM(50, activation='relu', return_sequences=False))
    lstm_autoencoder.add(RepeatVector(X_time_series.shape[1]))
    lstm_autoencoder.add(LSTM(50, activation='relu', return_sequences=True))
    lstm_autoencoder.add(LSTM(100, activation='relu', return_sequences=True))
    lstm_autoencoder.add(TimeDistributed(Dense(X_time_series.shape[2])))
    lstm_autoencoder.compile(optimizer='adam', loss='mse')
    
    lstm_history = lstm_autoencoder.fit(X_time_series, X_time_series, epochs=12, batch_size=batch_size, validation_split=0.1)
    lstm_autoencoder.save(f'lstm_autoencoder_model_{imeis_str}_{batch_size}.h5')

    print(f"LSTM Autoencoder - Batch Size: {batch_size}, Loss: {lstm_history.history['loss'][-1]}, Val Loss: {lstm_history.history['val_loss'][-1]}")


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
    print(f"training {imeis_str} ...")

    # Feature selection and preprocessing
    features = ['sound_db', 'noise_db', 'breath_rate', 'heart_rate', 'temperature', 'humedity', 'time_in_minutes']
    X = df[features]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test different batch sizes
    for batch_size in [32]: #[32, 64, 128, 256]:
        print(f"Testing batch size: {batch_size}")
        train_model(imeis_str, batch_size, X, X_scaled)
