import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos MNIST
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocesar las imágenes
X_train = X_train / 255.0
X_test = X_test / 255.0

# Redimensionar las imágenes a 28x28 píxeles
X_train = [cv2.resize(img, (28, 28)) for img in X_train]
X_test = [cv2.resize(img, (28, 28)) for img in X_test]

# Convertir las imágenes en arrays NumPy
X_train = np.array(X_train)
X_test = np.array(X_test)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Crear un modelo de red neuronal utilizando TensorFlow
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Precisión en el conjunto de prueba: {test_accuracy}')

# Hacer predicciones
predictions = model.predict(X_test)

# Puedes usar las predicciones para tomar decisiones basadas en la IA
