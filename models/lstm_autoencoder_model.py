import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
from sklearn.preprocessing import StandardScaler


class LSTMAutoencoderModel:
    def __init__(self, input_shape, learning_rate=0.001, patience=3):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.patience = patience
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(LSTM(100, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu', return_sequences=False))
        model.add(RepeatVector(self.input_shape[0]))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(100, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(self.input_shape[1])))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def train(self, X_time_series, batch_size, epochs=12, validation_split=0.1):
        early_stopping_loss = EarlyStopping(monitor='loss', patience=self.patience, restore_best_weights=True)
        early_stopping_val_loss = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        history = self.model.fit(
            X_time_series, X_time_series,
            epochs=epochs, batch_size=batch_size,
            validation_split=validation_split, verbose=1,
            callbacks=[early_stopping_loss, early_stopping_val_loss]
        )
        return history
    
    def save_model(self, file_path):
        self.model.save(file_path)
    
    def load_model(self, file_path):
        self.model = load_model(file_path)
    
    def predict(self, X, loss_cutoff):
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=(1, 2))
        anomalies = mse > loss_cutoff
        return anomalies, mse
    
    def evaluate_loss(self, X):
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=(1, 2))
        return mse
    
    def get_reconstruction_errors(self, X):
        reconstructions = self.model.predict(X)
        reconstruction_errors = np.abs(X - reconstructions)
        return reconstruction_errors
    
    def determine_loss_cutoff(self, mse, percentile=95):
        threshold = np.percentile(mse, percentile)
        return threshold

# Usage example
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
    batch_size=32

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


        # Reshape data for LSTM model
        X_time_series = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
        input_shape = (X_time_series.shape[1], X_time_series.shape[2])


        # Initialize and train the LSTM autoencoder model
        lstm_autoencoder = LSTMAutoencoderModel(input_shape=input_shape)
        history = lstm_autoencoder.train(X_time_series, batch_size, epochs=12)
        lstm_autoencoder.save_model(f'lstm_autoencoder_model_{imeis_str}_{batch_size}.keras')

        print(f"LSTM Autoencoder - Batch Size: {batch_size}, Loss: {history.history['loss'][-1]}, Val Loss: {history.history['val_loss'][-1]}")

        # Load the model for prediction
        lstm_autoencoder_loaded = LSTMAutoencoderModel(input_shape=input_shape)
        lstm_autoencoder_loaded.load_model(f'lstm_autoencoder_model_{imeis_str}_{batch_size}.keras')

        # Example prediction
        mse = lstm_autoencoder_loaded.evaluate_loss(X_time_series)
        loss_cutoff = lstm_autoencoder_loaded.determine_loss_cutoff(mse, percentile=95)
        print(f"Loss Cutoff: {loss_cutoff}")

        anomalies, mse = lstm_autoencoder_loaded.predict(X_time_series, loss_cutoff)
        print(f"Anomalies Detected: {anomalies}")

        # Get reconstruction errors
        reconstruction_errors = lstm_autoencoder_loaded.get_reconstruction_errors(X_time_series)
        print(f"Reconstruction Errors: {reconstruction_errors}")
