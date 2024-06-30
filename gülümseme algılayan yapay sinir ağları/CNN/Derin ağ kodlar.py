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

model_cnn_opt1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_cnn_opt1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history1_opt1 = model_cnn_opt1.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Performans değerlendirmesi
plt.plot(history1_opt1.history['loss'], label='Training loss')
plt.plot(history1_opt1.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss (Optimized)')


plt.plot(history1_opt1.history['accuracy'], label='Training accuracy')
plt.plot(history1_opt1.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy (Optimized)')
plt.legend()
plt.show()

model_cnn_opt2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_cnn_opt2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history2_opt2 = model_cnn_opt2.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

plt.plot(history2_opt2.history['loss'], label='Training loss')
plt.plot(history2_opt2.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss (Optimized)')


plt.plot(history2_opt2.history['accuracy'], label='Training accuracy')
plt.plot(history2_opt2.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy (Optimized)')
plt.legend()
plt.show()

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(X_train)

model_cnn_opt3 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_cnn_opt3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history3_opt3 = model_cnn_opt3.fit(datagen.flow(X_train, y_train, batch_size=32),
                                epochs=10, validation_data=(X_test, y_test))


plt.plot(history3_opt3.history['loss'], label='Training loss')
plt.plot(history3_opt3.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss (Optimized)')


plt.plot(history3_opt3.history['accuracy'], label='Training accuracy')
plt.plot(history3_opt3.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy (Optimized)')
plt.legend()
plt.show()

model_cnn_opt4 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model_cnn_opt4.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history4_opt4 = model_cnn_opt4.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

plt.plot(history4_opt4.history['loss'], label='Training loss')
plt.plot(history4_opt4.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss (Optimized)')


plt.plot(history4_opt4.history['accuracy'], label='Training accuracy')
plt.plot(history4_opt4.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy (Optimized)')
plt.legend()
plt.show()


