import tensorflow as tf
import tensorflow_datasets as tfds
import math
import matplotlib.pyplot as plt
datos,metadatos=tfds.load('fashion_mnist',as_supervised=True,with_info=True)

data_train, data_test= datos['train'], datos['test']

class_name=metadatos.features['label'].names

def normalizar(imagenes,etiquetas):
  imagenes=tf.cast(imagenes,tf.float32)
  imagenes/= 255 
  return imagenes,etiquetas

data_train=data_train.map(normalizar)
data_test== data_test.map(normalizar)

data_train=data_train.cache()
data_test = data_test.cache()


plt.figure(figsize=(10,10))
for i,(imagen,etiqueta) in enumerate (data_train.take(25)):
  imagen= imagen.numpy().reshape((28,28))
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(imagen,cmap=plt.cm.binary)
  plt.xlabel(class_name[etiqueta])
#plt.show()

modelo=tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28,1)),
  tf.keras.layers.Dense(50,activation=tf.nn.relu),
  tf.keras.layers.Dense(50,activation=tf.nn.relu),
  tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']

)

num_ej_entrenamiento=metadatos.splits["train"].num_examples
num_ej_pruebas=metadatos.splits["test"].num_examples
print(num_ej_entrenamiento)
print(num_ej_pruebas)
lote=32
data_train=data_train.repeat().shuffle(num_ej_entrenamiento).batch(lote)
data_test=data_test.batch(lote)

historial= modelo.fit(data_train,epochs=5,steps_per_epoch=math.ceil(num_ej_entrenamiento/lote))


import numpy as np

for imagenes_prueba, etiquetas_prueba in data_test.take(1):
  imagenes_prueba = imagenes_prueba.numpy()
  etiquetas_prueba = etiquetas_prueba.numpy()
  predicciones = modelo.predict(imagenes_prueba)

imagen = imagenes_prueba[6] 
imagen = np.array([imagen])
prediccion = modelo.predict(imagen)

print("Prediccion: " + class_name[np.argmax(prediccion[0])])