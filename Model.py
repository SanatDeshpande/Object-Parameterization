import keras
from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Reshape
from keras.models import Sequential
import numpy as np
import pickle
import matplotlib.pyplot as plt
#from DataFactory import save

with open("images", "rb") as f:
    images = np.asarray(pickle.load(f))
with open("vectors", "rb") as f:
    vectors = np.asarray(pickle.load(f))
vectors = vectors.reshape(vectors.shape[0], 60)
images = images.reshape(images.shape[0], 28, 28, 1)
model = Sequential()

model.add(Dense(100, input_shape=(60,), activation="relu"))
model.add(Reshape((10, 10, 1)))
model.add(Conv2DTranspose(80, 5, input_shape=(10, 10, 1), data_format="channels_last", activation="relu"))
model.add(BatchNormalization())

model.add(Conv2DTranspose(160, 6, data_format="channels_last", activation="relu"))
model.add(BatchNormalization())

model.add(Conv2DTranspose(80, 5, data_format="channels_last", activation="relu"))
model.add(BatchNormalization())

model.add(Conv2DTranspose(1, 6, data_format="channels_last", activation="relu"))
model.add(BatchNormalization())


model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(vectors, images, epochs=10, batch_size=16, validation_split=.1)

print(vectors.shape, vectors[0].shape)
sample = vectors[0].reshape((1, vectors[0].shape[0]))
prediction = model.predict(sample).reshape((28,28))
plt.imshow(prediction, cmap='gray')
plt.show()
