#!/usr/bin/env python
# coding: utf-8

# ## Red convolucinal con keras y mnist

# In[7]:


import keras
from numba import jit
import numba, time, sys
keras.__version__


# In[8]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[9]:


from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(10, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(2, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))
model.summary()


# In[10]:


from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print (train_images.shape)
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[11]:


import time

batch_size = 100
epochs = int(sys.argv[1])

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


# Method on gpu
@jit
def func_gpu():
	history=model.fit(train_images, train_labels,
		  batch_size=batch_size,
		  epochs=epochs,validation_data=(test_images, test_labels),
		  verbose=1)
		  
start = time.time()
func_gpu()
print("The time used to execute this is given below")
end = time.time()
print(end - start)


# Evaluación modelo

# In[6]:

'''
import matplotlib.pyplot as plt
ent_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(ent_loss) + 1)

plt.plot(epochs, ent_loss, 'b', label='Entrenamiento')
plt.plot(epochs, val_loss, 'r', label='Validación')
plt.title('Pérdida en Entrenamiento y Validación')
plt.xlabel('Epochs')
plt.ylabel('Pérdida')
plt.legend()

plt.show()


# In[8]:


import matplotlib.pyplot as plt

ent_loss = history.history['accuracy']
val_loss = history.history['val_accuracy']

epochs = range(1, len(ent_loss) + 1)

plt.plot(epochs, ent_loss, 'b', label='Entrenamiento')
plt.plot(epochs, val_loss, 'r', label='Validación')
plt.title('Pérdida en Entrenamiento y Validación')
plt.xlabel('Epochs')
plt.ylabel('Pérdida')
plt.legend()

plt.show()


# In[9]:


# Guardamos el modelo
model.save('mnistCNN10epochs.h5')


# In[ ]:

'''


