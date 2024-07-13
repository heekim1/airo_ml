from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np

class AutoencoderModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = Sequential()
        encoding_dim = 14
        hidden_dim = int(encoding_dim / 2)
        self.model.add(Dense(encoding_dim, activation="relu", input_shape=(input_dim,)))
        self.model.add(Dense(hidden_dim, activation="relu"))
        self.model.add(Dense(encoding_dim, activation="relu"))
        self.model.add(Dense(input_dim, activation="sigmoid"))
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X):
        self.model.fit(X, X, epochs=50, batch_size=32, validation_split=0.1)
        self.model.save('autoencoder_model.h5')

    def load(self, file_path='autoencoder_model.h5'):
        self.model = load_model(file_path)

    def predict(self, X, loss_cutoff):
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        anomalies = mse > loss_cutoff
        return anomalies

    def evaluate_loss(self, X):
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        return mse

    def get_reconstruction_errors(self, X):
        reconstructions = self.model.predict(X)
        reconstruction_errors = np.abs(X - reconstructions)
        return reconstruction_errors
