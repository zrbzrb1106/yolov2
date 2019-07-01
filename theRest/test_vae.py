"""
 This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 100
original_dim = 78*78
latent_dim = 4
intermediate_dim = 512
epochs = 200
epsilon_std = 1.0

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
vae = Model(x, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

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

x_train = x_train.astype(np.float16) / 255.
x_test = x_test.astype(np.float16) / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        validation_data=(x_test, None))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)
encoder_z = Model(x, z)

# vec = encoder_z.predict(x_test[0].reshape(1, 78*78))

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

n_show = 10
plt.figure(figsize=(10, 2), dpi=100)
vec = encoder_z.predict(x_test[:n_show])
res = generator.predict(vec)
for i in range(n_show):
    ax = plt.subplot(2, n_show, i + 1)
    plt.imshow((x_test[i]*255).astype(np.uint8).reshape(78,78))
    plt.gray()
    ax.set_axis_off()

    ax = plt.subplot(2, n_show, i + n_show + 1)
    plt.imshow((res[i]*255).astype(np.uint8).reshape(78,78))
    plt.gray()
    ax.set_axis_off()

plt.show()
