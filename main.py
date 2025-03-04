# Imported TensorFlow library
import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load training and test datasets from the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape and normalize the data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Create the model and add necessary layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

# Prepare the deep learning model for training
model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Start training the deep learning model
history1 = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate the model's performance on the test dataset and print the results
print(f'Model 1 - Test accuracy: {model.evaluate(test_images, test_labels)[1]}')

# Save the model
model.save('mnist_model.h5')

if os.path.exists('mnist_model.h5'):
    print("Model saved successfully!")
else:
    print("Error: Model not saved.")


# Display some examples from the training dataset to analyze model results
    
# plt.figure(figsize=(10, 10))
# for i in range(36):
#     plt.subplot(6, 6, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(str(train_labels[i]))
# plt.show()
