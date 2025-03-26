import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Betöltjük az MNIST adatbázist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalizáljuk az adatokat
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Modell definiálása (szekvenciális)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 2d kép lapítása 1d vektorrá
    tf.keras.layers.Dense(128, activation='relu'),  # rejtett réteg, 128 neuron relu aktivációval
    tf.keras.layers.Dropout(0.15),                   # dropout (overfitting ellen)
    tf.keras.layers.Dense(10, activation='softmax') # kimenet: 10 osztály, 10 számjegy
])

# 4. Modell fordítása
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Modell tanítása
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 6. Metrikák ábrázolása
plt.figure(figsize=(10, 4))

# Veszteség
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title('Loss over epochs')
plt.legend()

# Pontosság
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.title('Accuracy over epochs')
plt.legend()

# 7. Ábra mentése
plt.tight_layout()
plt.savefig("app/output.png")
model.save("models/model.h5")

