import argparse
import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Append the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.lstm_autoencoder_model import LSTMAutoencoderModel, create_sequences, data_generator

def main(input_csv, output_dir, sequence_length, batch_size):
    # Load data from CSV
    df = pd.read_csv(input_csv)

    # Assuming the DataFrame has a column 'imei' to identify rows by device
    imeis = df['imei'].unique()
    # Convert to string
    imeis_str = ', '.join(map(str, imeis))
    print(f"Training devices: {imeis_str} ...")

    # Feature selection and preprocessing
    features = ['sound_db', 'noise_db', 'breath_rate', 'heart_rate', 'temperature', 'humedity', 'time_in_minutes']
    X = df[features]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ensure no NaN or infinite values after scaling
    assert not np.any(np.isnan(X_scaled)), "Scaled data contains NaN values"
    assert not np.any(np.isinf(X_scaled)), "Scaled data contains infinite values"

    # Reshape data for LSTM model
    X_sequences = create_sequences(X_scaled, sequence_length)
    X_time_series = np.reshape(X_sequences, (X_sequences.shape[0], sequence_length, X_sequences.shape[2]))
    input_shape = (sequence_length, X_scaled.shape[1])

    # Initialize and train the LSTM autoencoder model
    lstm_autoencoder = LSTMAutoencoderModel(input_shape=input_shape)

    # Train the model
    history = lstm_autoencoder.train(X_time_series, batch_size=batch_size, epochs=50, validation_split=0.2)
    
    # Save the trained model
    model_path = os.path.join(output_dir, f'lstm_autoencoder_model_{imeis_str}_{batch_size}.keras')
    lstm_autoencoder.save_model(model_path)
    print(f"LSTM Autoencoder - Batch Size: {batch_size}, Loss: {history.history['loss'][-1]}, Val Loss: {history.history['val_loss'][-1]}")
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM Autoencoder using input CSV file.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the trained model (default: current directory)')
    parser.add_argument('--sequence_length', type=int, default=20, help='Length of the sequences for the LSTM model (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    args = parser.parse_args()
    main(args.input_csv, args.output_dir, args.sequence_length, args.batch_size)

