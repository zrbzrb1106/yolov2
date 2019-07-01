from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import cv2

# Deep Autoencoder

features_path = 'deep_autoe_features.pickle'
labels_path = 'deep_autoe_labels.pickle'

# this is the size of our encoded representations
encoding_dim = 16
input_length = 78 * 78

# this is our input placeholder; 784 = 28 x 28
input_img = Input(shape=(input_length, ))

my_epochs = 20

# "encoded" is the encoded representation of the inputs
encoded = Dense(encoding_dim * 16, activation='relu')(input_img)
encoded = Dense(encoding_dim * 8, activation='relu')(input_img)
encoded = Dense(encoding_dim * 4, activation='relu')(encoded)
encoded = Dense(encoding_dim * 2, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
decoded = Dense(encoding_dim * 4, activation='relu')(decoded)
decoded = Dense(encoding_dim * 8, activation='relu')(decoded)
decoded = Dense(encoding_dim * 16, activation='relu')(decoded)
decoded = Dense(input_length, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# Separate Encoder model

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# prepare input data
def load_fmaps_data():
    basedir = './channel_images/'
    ci_names = [basedir+i for i in os.listdir(basedir)]
    tmp = []
    for name in ci_names:
        if 'channel_20.jpg' in name:
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
    split_frac = 0.9
    train_num = int(num_total * split_frac)
    test_num = int(num_total * (1 - split_frac))
    x_train = data[0:train_num]
    x_test = data[train_num:]
    return (x_train, 1), (x_test, 1)

(x_train, _), (x_test, y_test) = load_fmaps_data()

# normalize all values between 0 and 1 and flatten the 28x28 images into vectors of size 784
x_train = x_train.astype(np.float16) / 255.
x_test = x_test.astype(np.float16) / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

# Train autoencoder for 50 epochs
# es = EarlyStopping(monitor='val_loss', mode='min', patience=3)
autoencoder.fit(x_train, x_train, epochs=my_epochs, batch_size=16, shuffle=True, validation_data=(x_test, x_test),
                callbacks=[], verbose=2)

# after 100 epochs the autoencoder seems to reach a stable train/test lost value

# Visualize the reconstructed encoded representations

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

n = 10  # how many digits we will display
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
