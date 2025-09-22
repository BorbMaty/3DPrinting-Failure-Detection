import tensorflow as tf
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 1. Betöltöm az MNIST adatbázist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalizáljuk az adatokat
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Modell definiálása
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # Bemeneti réteg 28x28-as képek 1D tömbbé alakítása
    tf.keras.layers.Dense(128, activation='relu'), # ReLU aktivációs függvény + 128 neuron a rejtett rétegben
    tf.keras.layers.Dropout(0.15), #Random megölöm a neuronok 15%-át, megelőzve az overfittinget
    tf.keras.layers.Dense(10, activation='softmax') # Kimeneti réteg 10 neuron, softmax aktivációs függvény
])

# 4. Modell fordítása
model.compile(optimizer='adam', # Adam optimalizáló
              loss='sparse_categorical_crossentropy', # Kereszthiba veszteség függvény
              metrics=['accuracy']) # Pontosság mint metrika

# 5. Modell tanítása
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 6. Timestamp generálása
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 7. Mappák létrehozása
os.makedirs("models", exist_ok=True) # itt lesznek a mentett modellek
os.makedirs("graphs", exist_ok=True) # itt lesznek a mentett grafikonok

# 8. Metrikák ábrázolása és mentése
plt.figure(figsize=(10, 4)) # Grafikon méretezése

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

plt.tight_layout()
plt.savefig(f"graphs/metrics_{timestamp}.png")

# 9. Modell mentése timestamp-es névvel
model.save(f"models/model_{timestamp}.keras")
