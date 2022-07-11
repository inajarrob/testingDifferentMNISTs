#!/usr/bin/env python
# coding: utf-8
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras.datasets import mnist
from keras import utils
from numba import jit
import numba, time, sys
from keras.layers import Dense,Dropout # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/



# Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training data shape: ", x_train.shape) # (60000, 28, 28) -- 60000 images, each 28x28 pixels
print("Test data shape", x_test.shape) # (10000, 28, 28) -- 10000 images, each 28x28

# Flatten the images
image_vector_size = 28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

# Convert to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
print("First 5 training lables as one-hot encoded vectors:\n", y_train[:10])


# In[3]:
print(x_train.ndim) 
print(x_train.shape)
print(x_test.ndim) 
print(x_test.shape)


# In[4]:
model = Sequential()
model.add(Dense(units=100, input_shape=(784,),activation='relu'))
#model.add(Dense(units=200, activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))
model.summary()

model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Method on gpu
@jit
def func_gpu():
    history = model.fit(x_train, y_train, 
                    batch_size=100, 
                    epochs=int(sys.argv[1]), 
                    validation_data=(x_test, y_test))

start = time.time()
func_gpu()
print("The time used to execute this is given below")
end = time.time()
print("GPU: ", str(end - start))


'''
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

loss, accuracy  = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')


# In[7]:


# Guardamos el modelo
model.save('mnistANN15epochsGPU.h5')
'''

# In[ ]:




