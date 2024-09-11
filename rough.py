# import numpy as np
# from scipy.io import wavfile

# # Load the audio file
# sample_rate, audio_data = wavfile.read(r'C:\Users\rohit\OneDrive\Desktop\audio dataset\glass jug.wav')

# # Check the dimensions of the audio data
# print(audio_data.shape)

#backup
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
# Define the directories containing the audio files
full_dir = r'C:\Users\rohit\OneDrive\Desktop\water_level_percentage\clean_audio_files\empty'
empty_dir = r'C:\Users\rohit\OneDrive\Desktop\water_level_percentage\clean_audio_files\full'
half_dir = r'C:\Users\rohit\OneDrive\Desktop\water_level_percentage\clean_audio_files\half-full'

# Define the parameters for feature extraction
sr = 22050
n_fft = 2048
hop_length = 512
n_mfcc = 13

# Define a function to extract features from an audio file
def extract_features(file_path,sample_rate):
    # Load the audio file and extract its features
    y, sr = librosa.load(file_path, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    # Compute the mean and standard deviation of each feature
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    # Concatenate the mean and standard deviation vectors into a single feature vector
    features = np.concatenate((mfcc_mean, mfcc_std))
    return features

# Define a function to load and preprocess the audio dataset
def load_dataset():
    X = []
    y = []
    # Load the full container audio files
    for filename in os.listdir(full_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(full_dir, filename)
            features = extract_features(file_path,sample_rate=sr)
            X.append(features)
            y.append(1.0)
    # Load the empty container audio files
    for filename in os.listdir(empty_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(empty_dir, filename)
            features = extract_features(file_path,sample_rate=sr)
            X.append(features)
            y.append(0.0)
    # Load the half-full container audio files
    for filename in os.listdir(half_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(half_dir, filename)
            features = extract_features(file_path,sample_rate=sr)
            X.append(features)
            y.append(0.5)
    # Convert the feature and label lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y

# Load and preprocess the audio dataset
X, y = load_dataset()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(26,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training set
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the testing set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

# Use the model to predict the percentage of water level in a container given an audio input
file_path = r'C:\Users\rohit\OneDrive\Desktop\water_level_percentage\clean_audio_files\full\half to full plastic tub.wav'
features = extract_features(file_path,sample_rate=sr)
print(features.shape)
prediction = model.predict(features.reshape(1,-1))
# Print the predicted percentage of water level
print('Predicted percentage of water level:', prediction[0][0] * 100)
