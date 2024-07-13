from keras.models import Sequential, load_model
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import numpy as np
import matplotlib.pyplot as plt

class LSTMAutoencoderModel:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(100, activation='relu', input_shape=input_shape, return_sequences=True))
        self.model.add(LSTM(50, activation='relu', return_sequences=False))
        self.model.add(RepeatVector(input_shape[0]))
        self.model.add(LSTM(50, activation='relu', return_sequences=True))
        self.model.add(LSTM(100, activation='relu', return_sequences=True))
        self.model.add(TimeDistributed(Dense(input_shape[1])))
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X):
        self.model.fit(X, X, epochs=50, batch_size=32, validation_split=0.1)
        self.model.save('lstm_autoencoder_model.h5')

    def load(self, file_path='lstm_autoencoder_model.h5'):
        self.model = load_model(file_path)

    def predict(self, X, loss_cutoff):
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=(1, 2))
        anomalies = mse > loss_cutoff
        return anomalies
    
    def evaluate_loss(self, X):
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=(1, 2))
        return mse

    def get_reconstruction_errors(self, X):
        reconstructions = self.model.predict(X)
        reconstruction_errors = np.abs(X - reconstructions)
        return reconstruction_errors

    """
    def predict(self, X, loss_cutoff):
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=(1, 2))
        anomalies = mse > loss_cutoff
        return anomalies
    """

# Function to determine the loss cutoff threshold
def determine_loss_cutoff(mse, percentile=95):
    threshold = np.percentile(mse, percentile)
    return threshold