import tensorflow as tf
from tensorflow.keras.datasets import mnist  # Ejemplo de importación
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np

# Define una función de activación personalizada que utiliza la función softmax.
def softmax_v2(x):
    return tf.nn.softmax(x)

# Carga y preprocesa los datos del conjunto de datos MNIST.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normaliza los datos de entrenamiento y prueba dividiéndolos por 255.0.
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define el modelo secuencial.
model = Sequential([
    # Aplana las imágenes de 28x28 píxeles a un vector de 784 elementos.
    Flatten(input_shape=(28, 28)),
    # Capa densa con 128 unidades y función de activación ReLU.
    Dense(128, activation='relu'),
    # Otra capa densa con 128 unidades y función de activación ReLU.
    Dense(128, activation='relu'),
    # Capa de salida con 10 unidades (correspondientes a las 10 clases de dígitos) y la función de activación personalizada softmax.
    Dense(10, activation=softmax_v2)
])

# Compila el modelo especificando el optimizador, la función de pérdida y las métricas de evaluación.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrena el modelo con los datos de entrenamiento durante 3 épocas.
model.fit(x_train, y_train, epochs=3)

# Evalúa el modelo con los datos de prueba para obtener la pérdida y la precisión.
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)  # Muestra la pérdida del modelo.
print(val_acc)  # Muestra la precisión del modelo.

# Muestra la primera imagen del conjunto de entrenamiento.
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()
print(x_train[0])  # Muestra los valores de los píxeles de la imagen.

# Guarda el modelo entrenado en un archivo con la extensión .keras.
model.save('epic_num_reader.keras')

# Carga el modelo guardado, especificando la función de activación personalizada en custom_objects.
custom_objects = {'softmax_v2': softmax_v2}
new_model = tf.keras.models.load_model('epic_num_reader.keras', custom_objects=custom_objects)

# Realiza predicciones con el modelo cargado utilizando los datos de prueba.
predictions = new_model.predict(x_test)
print(predictions)  # Muestra las predicciones para el primer lote de datos de prueba.

# Muestra la clase predicha para la primera imagen del conjunto de prueba.
print(np.argmax(predictions[0]))

# Muestra la primera imagen del conjunto de prueba.
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()
