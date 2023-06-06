import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Define the Pure CNN model
model = tf.keras.Sequential([
    layers.Conv2D(8, (3, 3), activation="relu", input_shape=(28, 28, 1)),  # 28x28x1 -> 26x26x8
    layers.Conv2D(16, (3, 3), activation="relu"),  # 26x26x8 -> 24x24x16
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Training loop
while True:
    history = model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1)
    _, validation_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"Validation Accuracy: {validation_accuracy*100:.2f}%")
    
    # Stop training if the validation accuracy reaches the desired threshold
    if validation_accuracy >= 0.994:
        break
