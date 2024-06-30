import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

positive_path = '/content/drive/MyDrive/yapayodev/SMILEs/positives/positives7'
negative_path = '/content/drive/MyDrive/yapayodev/SMILEs/negatives/negatives7'

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            images.append(img)
    return images

positive_images = load_images_from_folder(positive_path)
negative_images = load_images_from_folder(negative_path)

X = np.array(positive_images + negative_images)
y = np.array([1] * len(positive_images) + [0] * len(negative_images))

# Normalize images
X = X / 255.0

# Reshape for CNN input
X = X.reshape(X.shape[0], 64, 64, 1)

# test ve eğitim verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma
model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Model özeti
model_cnn.summary()

model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model eğitimi
history_cnn = model_cnn.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Eğitim ve doğrulama kaybı
plt.plot(history_cnn.history['loss'], label='Training loss')
plt.plot(history_cnn.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Eğitim ve doğrulama doğruluğu
plt.plot(history_cnn.history['accuracy'], label='Training accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()







