import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Define directories
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(app_dir, "model", "saved_model")
model_path = os.path.join(model_dir, "mnist_cnn.h5")

# Prepare the MNIST data
def load_and_prepare_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    return x_train, y_train, x_test, y_test

# Define the CNN model
def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    x_train, y_train, x_test, y_test = load_and_prepare_data()
    model = build_model()
    model.summary()
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_path)
    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    main()