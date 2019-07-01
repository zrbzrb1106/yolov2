from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import cv2

input_shape = (78, 78, 1)
input_img = Input(shape=input_shape)    # adapt this if using 'channels_first' image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8), i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

def load_fmaps_data():
    basedir = './channel_images/'
    ci_names = [basedir+i for i in os.listdir(basedir)]
    tmp = []
    for name in ci_names:
        if 'channel_10.jpg' in name:
            tmp.append(name)
    ci_names = tmp
    data = []
    for idx, name in enumerate(ci_names):
        if idx % 1000 == 0:
            print("{} finished".format(idx))
        data.append(cv2.imread(name, 0))
    data = np.array(data)
    np.random.shuffle(data)
    num_total = data.shape[0]
    split_frac = 0.8
    train_num = int(num_total * split_frac)
    test_num = int(num_total * (1 - split_frac))
    x_train = data[0:train_num]
    x_test = data[train_num:]
    return (x_train, 1), (x_test, 1)

(x_train, _), (x_test, y_test) = load_fmaps_data()

x_train = x_train.astype(np.float16) / 255.
x_test = x_test.astype(np.float16) / 255.
x_train = np.reshape(x_train, (len(x_train), 78, 78, 1))    # adapt this if using 'channels_first' image data format
x_test = np.reshape(x_test, (len(x_test), 78, 78, 1))       # adapt this if using 'channels_first' image data format

# open a terminal and start TensorBoard to read logs in the autoencoder subdirectory
# tensorboard --logdir=autoencoder
es = EarlyStopping(monitor='val_loss', mode='min', patience=3)
autoencoder.fit(x_train, x_train, epochs=100, batch_size=64, shuffle=True, validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='conv_autoencoder'), es], verbose=2)

# take a look at the reconstructed digits
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(10, 2), dpi=100)
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow((x_test[i]*255).astype(np.uint8).reshape(78,78))
    plt.gray()
    ax.set_axis_off()

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow((decoded_imgs[i]*255).astype(np.uint8).reshape(78, 78))
    plt.gray()
    ax.set_axis_off()
plt.show()

K.clear_session()
