import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from tensorflow.keras.layers import LSTM, TimeDistributed, Reshape

positive_path = '/content/drive/MyDrive/yapayodev/SMILEs/positives/positives7'
negative_path = '/content/drive/MyDrive/yapayodev/SMILEs/negatives/negatives7'

# Veriyi RNN/LSTM için yeniden şekillendirme
X_train_lstm = X_train.reshape((X_train.shape[0], 64, 64))
X_test_lstm = X_test.reshape((X_test.shape[0], 64, 64))

# test ve eğitim verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma
model_lstm = Sequential([
    LSTM(50, input_shape=(64, 64)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Model özeti
model_lstm.summary()

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_lstm = model_lstm.fit(X_train_lstm, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Performans değerlendirmesi
plt.plot(history_lstm.history['loss'], label='Training loss')
plt.plot(history_lstm.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss')

plt.plot(history_lstm.history['accuracy'], label='Training accuracy')
plt.plot(history_lstm.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

