import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
from sklearn.preprocessing import StandardScaler

class AutoencoderModel:
    def __init__(self, input_dim, encoding_dim=14, hidden_dim=None, learning_rate=0.001, patience=3):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else int(encoding_dim / 2)
        self.learning_rate = learning_rate
        self.patience = patience
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))
        model.add(Dense(self.encoding_dim, activation="relu"))
        model.add(Dense(self.hidden_dim, activation="relu"))
        model.add(Dense(self.encoding_dim, activation="relu"))
        model.add(Dense(self.input_dim, activation="sigmoid"))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def train(self, X_scaled, batch_size, epochs=12, validation_split=0.1):
        early_stopping_loss = EarlyStopping(monitor='loss', patience=self.patience, restore_best_weights=True)
        early_stopping_val_loss = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs, batch_size=batch_size,
            validation_split=validation_split, verbose=1,
            callbacks=[early_stopping_loss, early_stopping_val_loss]
        )
        return history
    
    def save_model(self, file_path):
        self.model.save(file_path)
    
    def load_model(self, file_path):
        self.model = load_model(file_path, custom_objects={'MeanSquaredError': tf.keras.losses.MeanSquaredError()})
    
    def predict(self, X, loss_cutoff):
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        anomalies = mse > loss_cutoff
        return anomalies, mse
    
    def evaluate_loss(self, X):
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        return mse
    
    def get_reconstruction_errors(self, X):
        reconstructions = self.model.predict(X)
        reconstruction_errors = np.abs(X - reconstructions)
        return reconstruction_errors
    
    def determine_loss_cutoff(self, mse, percentile=95):
        threshold = np.percentile(mse, percentile)
        return threshold

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

        input_dim = X_scaled.shape[1]

        # Path to the saved model
        #model_path = f'/Users/hkim75/Airo/airo_ml/train/model_3_single_07132024/autoencoder_model_108691961563855_32.h5'

        # Initialize and train the autoencoder model
        autoencoder = AutoencoderModel(input_dim=input_dim)
        history = autoencoder.train(X_scaled, batch_size, epochs=12)
        autoencoder.save_model(f'autoencoder_model_{imeis_str}_{batch_size}.keras')

        print(f"Autoencoder - Batch Size: {batch_size}, Loss: {history.history['loss'][-1]}, Val Loss: {history.history['val_loss'][-1]}")
        
        
        # Load the model for prediction
        autoencoder_loaded = AutoencoderModel(input_dim=input_dim)
        autoencoder_loaded.load_model(f'autoencoder_model_{imeis_str}_{batch_size}.keras')

        # Example prediction
        mse = autoencoder_loaded.evaluate_loss(X_scaled)
        loss_cutoff = autoencoder_loaded.determine_loss_cutoff(mse, percentile=95)
        print(f"Loss Cutoff: {loss_cutoff}")

        anomalies, mse = autoencoder_loaded.predict(X_scaled, loss_cutoff)
        print(f"Anomalies Detected: {anomalies}")

        # Get reconstruction errors
        reconstruction_errors = autoencoder_loaded.get_reconstruction_errors(X_scaled)
        print(f"Reconstruction Errors: {reconstruction_errors}")
