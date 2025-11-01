# Image-Colorization-Using-CNN
Enhancements in this Version 
from google.colab import files
uploaded = files.upload()

import cv2
import numpy as np
import matplotlib.pyplot as plt

image_list = []

# Load and resize uploaded images to 32x32
for filename in uploaded.keys():
    img = cv2.imread(filename)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (32, 32))             # Resize to 32x32
    image_list.append(img)

# Convert to NumPy array
x_train = np.array(image_list).astype('float32') / 255.0

# Convert to grayscale
x_train_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_train])
x_train_gray = x_train_gray.reshape(-1, 32, 32, 1)

print("Loaded images:", x_train.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam

model = Sequential()

# Encoder
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))

# Decoder
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))

# Output layer
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

model.compile(optimizer=Adam(), loss='mse')
model.summary()
model.fit(x_train_gray, x_train, epochs=100, batch_size=2, verbose=1)

# Predict colorized output
output = model.predict(x_train_gray)

# Display grayscale, original color, and predicted color
plt.figure(figsize=(15, 5))
for i in range(len(x_train)):
    # Grayscale input
    plt.subplot(3, len(x_train), i + 1)
    plt.imshow(x_train_gray[i].reshape(32, 32), cmap='gray')
    plt.axis('off')
    plt.subplot(3, len(x_train), i + 1 + len(x_train))
    plt.imshow(x_train[i])
    plt.axis('off')


plt.suptitle("Top: Grayscale | Middle: Original | ", fontsize=14)
plt.show()
