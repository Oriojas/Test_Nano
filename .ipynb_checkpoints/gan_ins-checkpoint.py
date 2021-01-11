# -*- coding: utf-8 -*-
"""GAN_INS.ipynb

# Generación de rostros de insectos a traves de una red generadora antagónica
## Librerias a utilizar
"""

import os
from imageio import imread, imwrite
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization, Conv2D, LeakyReLU, Flatten, Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.utils as ut
from random import randint

# Semilla de los numeros aleatoria para poder comparar resultados
np.random.seed(137)

"""## Doy el acceso a los directorios de trabajo"""

dataset = "Data_Ins" # carpeta donde guardo las imagenes verdaderas de entrenamiento
ejemplos = "Ejemplo_Ins"

"""## Estas son las utilidades del proyecto"""

# Creación del set de entrenamiento
def cargar_datos():
	print('Creando set de entrenamiento...',end="",flush=True)
	filelist = os.listdir(dataset) # la variable donde estan las carpetas con las imagenes

	n_imgs = len(filelist)
	x_train = np.zeros((n_imgs,128,128,3)) # esta parte del codigo carga las imagenes de 128 x 128 pixeles y 3 canales rgb

	for i, fname in enumerate(filelist):
		if fname != '.DS_Store':
			imagen = imread(os.path.join(dataset,fname))
			x_train[i,:] = (imagen - 127.5)/127.5 # las imagenes se normalizan para estar en un rango de -1 hasta 1
	print('¡Listo!')

	return x_train

# Visualizar imágenes del set de entrenamiento
def visualizar_imagen(nimagen,x_train):
	imagen = (x_train[nimagen, :]*127.5) + 127.5
	imagen = np.ndarray.astype(imagen, np.uint8)
	plt.imshow(imagen.reshape(128,128,3))
	plt.axis('off')
	plt.show()

# Visualización de algunas imagenes obtenidas con el generador
def graficar_imagenes_generadas(epoch, generador, ejemplos=16, dim=(4,4), figsize=(10,10)):
    ruido = np.random.normal(0,1,[ejemplos,100])
    imagenes_generadas = generador.predict(ruido)
    imagenes_generadas.reshape(ejemplos,128,128,3)
    imagenes_generadas = imagenes_generadas*127.5 + 127.5
    plt.figure(figsize=figsize)
    for i in range(ejemplos):
        plt.subplot(dim[0],dim[1], i+1)
        plt.imshow(imagenes_generadas[i].astype('uint8'), interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("Ejemplo_Ins/GAN_imagen_generada_%d.png" %epoch)
    plt.close()

# Generar imágenes ejemplo
def generar_imagenes(generador,nimagenes):
	ruido = np.random.normal(0,1,[nimagenes,100])
	imagenes_generadas = generador.predict(ruido)
	imagenes_generadas.reshape(nimagenes,128,128,3)
	imagenes_generadas = imagenes_generadas*127.5 + 127.5
	imagenes_generadas.astype('uint8')
	for i in range(nimagenes):
		imwrite(os.path.join(ejemplos,'ejemplo_'+str(i)+'.png'),imagenes_generadas[i].reshape(128,128,3)) # guardo las imagenes

"""## Parámetros de las redes neuronales"""

# Inicialización de parámetros
OPTIMIZADOR = Adam(lr = 0.0001, beta_1 = 0.5) # optimisador que se una variable del gradiente desendente que es estocástico lo que permite una convergencia más rápida
TAM_ENTRADA = 100 # tamaño del vector de entrada que generará los primeros rostros
ERROR = 'binary_crossentropy'# error a utilizar es la entropia cruzada
LEAKY_SLOPE = 0.2 # recta perfecta debo estudiar mas este parámetro
TAM_LOTE = 15 # tamaño del lote de las imagenes que se le pasaran al los modelos
N_ITS = 1000 # cantidad de iteraciones del modelo

# Crear set de entrenamiento y visualizar una imagen
x_train = cargar_datos()
IMG_RAN = randint(1, len(x_train)) # cargar un numero aleatorio de imagen
visualizar_imagen(IMG_RAN, x_train) # el numero de la imagen que quiero ver

"""## Red generadora"""

# Generador
def crear_generador():
    modelo = Sequential()
    modelo.add(Dense(1024*4*4, use_bias=False, input_shape=(TAM_ENTRADA,))) # capa neuronal de 1024*4*4
    modelo.add(BatchNormalization(momentum=0.3))
    modelo.add(LeakyReLU(alpha=LEAKY_SLOPE))
    modelo.add(Reshape((4,4,1024))) # se redimenciona la capa a 4*4*1024
    #4x4x1024

    modelo.add(Conv2DTranspose(512,(5,5),strides=(2,2),padding='same', use_bias=False)) 
    modelo.add(BatchNormalization(momentum=0.3)) # normnalizo los valores para mejorar la convergencia del módelo los valores tendran media 0 y deviación estandar de 1
    modelo.add(LeakyReLU(alpha=LEAKY_SLOPE)) # funciòn de activación
    #8x8x512 convolución inversa 1

    modelo.add(Conv2DTranspose(256,(5,5),strides=(2,2),padding='same', use_bias=False))
    modelo.add(BatchNormalization(momentum=0.3)) # normnalizo los valores para mejorar la convergencia del módelo los valores tendran media 0 y deviación estandar de 1
    modelo.add(LeakyReLU(alpha=LEAKY_SLOPE)) # funciòn de activación
    #16x16x256 convolución inversa 2

    modelo.add(Conv2DTranspose(128,(5,5),strides=(2,2),padding='same', use_bias=False))
    modelo.add(BatchNormalization(momentum=0.3)) # normnalizo los valores para mejorar la convergencia del módelo los valores tendran media 0 y deviación estandar de 1
    modelo.add(LeakyReLU(alpha=LEAKY_SLOPE)) # funciòn de activación
    #32x32x128 convolución inversa 3

    modelo.add(Conv2DTranspose(64,(5,5),strides=(2,2),padding='same', use_bias=False))
    modelo.add(BatchNormalization(momentum=0.3))# normnalizo los valores para mejorar la convergencia del módelo los valores tendran media 0 y deviación estandar de 1
    modelo.add(LeakyReLU(alpha=LEAKY_SLOPE)) # funciòn de activación
    #64x64x64 convolución inversa 4

    modelo.add(Conv2DTranspose(3, (5,5),strides=(2,2),padding='same', use_bias=False))
    modelo.add(Activation('tanh')) # la tgth me permite que los pixeles de sálida esten en el rango de -1 a 1 como se efectuo al cargar el dataset de entrenamiento
    #128x128x3 convolución inversa final al tamaño deseado

    modelo.compile(optimizer=OPTIMIZADOR, loss=ERROR)

    return modelo

generador = crear_generador()
print("Generador creado")
#generador.summary()

"""## Discriminador"""

# Discriminador (el inverso del generador)
def crear_discriminador():
    modelo = Sequential()
    modelo.add(Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=(128,128,3), # la imagen de netrada de 128*128*3 como se termino en el generador
        use_bias=False))
    modelo.add(LeakyReLU(alpha=LEAKY_SLOPE))
    #64x64x64

    modelo.add(Conv2D(128, (5,5), strides=(2,2), padding='same', use_bias=False)) # progresivamente extrae las caracteristicas de las imagenes de entrada
    modelo.add(BatchNormalization(momentum=0.3))
    modelo.add(LeakyReLU(alpha=LEAKY_SLOPE)) # funciòn de activación
    #32x32x128

    modelo.add(Conv2D(256, (5,5), strides=(2,2), padding='same', use_bias=False)) # progresivamente extrae las caracteristicas de las imagenes de entrada
    modelo.add(BatchNormalization(momentum=0.3))
    modelo.add(LeakyReLU(alpha=LEAKY_SLOPE)) # funciòn de activación
    #16x16x256

    modelo.add(Conv2D(512, (5,5), strides=(2,2), padding='same', use_bias=False))# progresivamente extrae las caracteristicas de las imagenes de entrada
    modelo.add(BatchNormalization(momentum=0.3))
    modelo.add(LeakyReLU(alpha=LEAKY_SLOPE)) # funciòn de activación
    #8x8x512

    modelo.add(Conv2D(1024, (5,5), strides=(2,2), padding='same', use_bias=False)) # progresivamente extrae las caracteristicas de las imagenes de entrada
    modelo.add(BatchNormalization(momentum=0.3))
    modelo.add(LeakyReLU(alpha=LEAKY_SLOPE)) # funciòn de activación
    #4x4x1024

    modelo.add(Flatten())
    modelo.add(Dense(1, activation='sigmoid', use_bias=False)) # al final el numero indicara la categoria de la imagen 1 = verdadera o 2 = falsa

    modelo.compile(optimizer=OPTIMIZADOR, loss=ERROR)

    return modelo

discriminador = crear_discriminador()
print("Discriminador creado")
#discriminador.summary()

"""## Entrenamiento de las dos redes"""

# GAN
def crear_GAN(generador, discriminador):
    modelo = Sequential() # creo el contenedor donde se ejecutarán los modelos
    modelo.add(generador) # GENERADOR
    discriminador.trainable = False
    modelo.add(discriminador) # DISCRIMINADOR
    modelo.compile(optimizer = OPTIMIZADOR, loss = ERROR)

    return modelo

gan = crear_GAN(generador, discriminador)
gan.summary()

# Entrenamiento
n_lotes = x_train.shape[0]/TAM_LOTE  # tamaño del lote de las imagenes que se le pasaran al los modelos

for i in range(1,N_ITS+1):
    print("Epoch " + str(i))

    # Crear un "batch" de imágenes falsas y otro con imágenes reales
    # se crean dos lotes de imagenes de 128
    ruido = np.random.normal(0,1,[TAM_LOTE,TAM_ENTRADA]) # lote 1 imagenes reales
    batch_falsas = generador.predict(ruido)

    idx = np.random.randint(low=0, high=x_train.shape[0],size=TAM_LOTE) # lote 2 imagenes falsas obtenidas con el generador
    batch_reales = x_train[idx]

    # Entrenar discriminador con imagener falsas y reales, y en cada
    # caso calcular el error
    discriminador.trainable = True # se descogelan los valores del discriminador o lo contrario

    dError_reales = discriminador.train_on_batch(batch_reales, # entreno el discriminador con los datos del generador
        np.ones(TAM_LOTE)*0.9) # creo la categoria verdadero 1
    dError_falsas = discriminador.train_on_batch(batch_falsas,
        np.zeros(TAM_LOTE)*0.1) # creo la categoria falsa 2

    discriminador.trainable = False # se congelan los coeficientes

    # Entrenar GAN: se generará ruido aleatorio y se presentará a la GAN
    # como si fuesen imagenes reales
    ruido = np.random.normal(0,1,[TAM_LOTE,TAM_ENTRADA]) # vector de rango aleatorio para generar la imagen falsa
    gError = gan.train_on_batch(ruido, np.ones(TAM_LOTE))

    # Graficar ejemplo de imágenes generadas, cada 100 iteraciones
    if i==1 or i%500 == 0: # cada 500 iteraciones grafico las imagenes
        graficar_imagenes_generadas(i,generador) # guardar los pesos del modelo
        generador.save("generador.h5")
        
generar_imagenes(generador,100)

# serializar el modelo a JSON
generador_json = generador.to_json()
with open("generador.json", "w") as json_file:
    json_file.write(generador_json)
print("Grafo guardado")

