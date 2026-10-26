import tensorflow as tf
from tensorflow import keras

# Load the MNIST handwritten digit dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values from 0–255 to 0–1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Configure the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
model.save("digit_model.keras")
print("Model saved successfully!")

# Test predictions on 5 sample images
predictions = model.predict(x_test[:5], verbose=0)

for i in range(5):
    predicted_digit = predictions[i].argmax()
    actual_digit = y_test[i]

    print(
        f"Actual: {actual_digit} | "
        f"Predicted: {predicted_digit}"
    )