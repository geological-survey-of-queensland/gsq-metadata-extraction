import tensorflow as tf
import numpy as np

def main():

    # mnist = tf.keras.datasets.mnist
    data = np.load("mnist.npz")
    x_train, y_train, x_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]

    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=2)
    model.evaluate(x_test, y_test)

if __name__ == "__main__": main()