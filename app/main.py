import tensorflow as tf
import matplotlib.pyplot as plt

# Download and load in the MNIST database
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Show one element from the dataset
plt.imshow(x_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")

# Save the plot to a file in the app directory
plt.savefig("output.png")

