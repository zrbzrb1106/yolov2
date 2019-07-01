from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import cv2
import random

# Single fully-connected neural layer as encoder and decoder

use_regularizer = True
my_regularizer = None
my_epochs = 100
features_path = 'simple_autoe_features.pickle'
labels_path = 'simple_autoe_labels.pickle'

if use_regularizer:
    # add a sparsity constraint on the encoded representations
    # note use of 10e-5 leads to blurred results
    my_regularizer = regularizers.l1(10e-8)
    # and a larger number of epochs as the added regularization the model
    # is less likely to overfit and can be trained longer
    my_epochs = 100
    features_path = 'sparse_autoe_features.pickle'
    labels_path = 'sparse_autoe_labels.pickle'

# this is the size of our encoded representations
encoding_dim = 512   # 32 floats -> compression factor 24.5, assuming the input is 784 floats

# this is our input placeholder; 784 = 28 x 28
input_shape = 78 * 78
input_img = Input(shape=(input_shape, ))

# "encoded" is the encoded representation of the inputs
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=my_regularizer)(input_img)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_shape, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# Separate Encoder model

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# Separate Decoder model

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Train to reconstruct MNIST digits

# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

# prepare input data
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
es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
history = autoencoder.fit(x_train, x_train, epochs=my_epochs, batch_size=256, shuffle=True, validation_data=(x_test, x_test),
                verbose=2, callbacks=[es])

# after 50/100 epochs the autoencoder seems to reach a stable train/test lost value

# Visualize the reconstructed encoded representations

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# save latent space features 32-d vector
# pickle.dump(encoded_imgs, open(features_path, 'wb'))
# pickle.dump(y_test, open(labels_path, 'wb'))


def plot_history_loss():
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('C:\\d\\exercise\\da\\yolov2\\dev\\loss_histroy.png', format='png')
    plt.close()
plot_history_loss()

n = 10  # how many digits we will display
plt.figure(figsize=(10, 2), dpi=100)
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow((x_test[i]*255).astype(np.uint8).reshape(78, 78))
    plt.gray()
    ax.set_axis_off()

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow((decoded_imgs[i]*255).astype(np.uint8).reshape(78, 78))
    plt.gray()
    ax.set_axis_off()

plt.show()

K.clear_session()
