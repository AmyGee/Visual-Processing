from keras.datasets import cifar10, mnist
from matplotlib import pyplot as plt
from scipy.misc import toimage
import numpy as np
from keras.datasets import cifar10
from keras.layers import *
from keras.models import Model
from keras.constraints import maxnorm
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
from keras.callbacks import TensorBoard
from autoencoder import *
from utils import *
from experiments import*
import keras

#DICE version 07/04

#Following the tutorial at https://blog.keras.io/building-autoencoders-in-keras.html & including comments from gregorygunderson.com & jeremyjordan.me

def run_variational(fname, save = False, save_name = None):

    epochs = 1
    batch_size = 100
    imgs = load(fname)
    imgs = np.array(imgs)
    x_train, x_test = split_into_test_train(imgs)

    input_shape = x_train.shape
    intermediate_dim = 400
    latent_dim = 2
    epsilon_std = 1.0
    #First we create the encoder network which maps inputs to our latent distribution parameters:

    x = Input(shape = (100, 100, 3))
    print(x)
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h) #Use log variance instead of standard deviation as it is more convenient and helps with numerical stability

    #Use the latent distribution parameters to sample new & similar points from the latent space

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std) #Normally distributed values in a tensor to be used as noise
        return z_mean + K.exp(z_log_sigma) * epsilon #Reparameterization happens here!
    #Shifting the random sample by the mean and scaling it by the variance

    #Make the sampling the input
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma]) #Lambda wraps an arbitrary expression as a layer, so here we are wrapping the sampling (latent space) as our input layer

    #Map the latent points to reconstructued inputs

    decoder_h = Dense(intermediate_dim, activation='relu')
    print (decoder_h)
    decoder_mean = Dense(batch_size, activation='sigmoid')
    print (decoder_mean)
    h_decoded = decoder_h(z)
    print (h_decoded)
    x_decoded_mean = decoder_mean(h_decoded)
    print (x_decoded_mean)

    #Instantiate 3 models

    #End-to-end autoencoder for mapping inputs to reconstructions

    vae = Model(x, x_decoded_mean)

    #Encoder mapping from inputs to latent space
    encoder = Model(x, z_mean)

    #Generator which takes points from the latent space to output the reconstructed samples

    decoder_input = Input(shape=(latent_dim,))
    print (decoder_input)
    _h_decoded = decoder_h(decoder_input)
    print (_h_decoded)
    _x_decoded_mean = decoder_mean(_h_decoded) #Push z through decoder
    print (_x_decoded_mean)
    generator = Model(decoder_input, _x_decoded_mean)

    #Train using the end-to-end model with a custom loss & K-L divergence regularization

    def vae_loss(x, x_decoded_mean):
        xent_loss = metrics.binary_crossentropy(x, x_decoded_mean) #Reconstruction loss
        #Binary crossentropy because the decoding term is a Bernoulli multi layered perceptron - is it worth also trying Gaussian + MSE??
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1) #Variational loss
        return xent_loss + kl_loss #Combine the losses

    vae.compile(optimizer='rmsprop', loss=vae_loss)

    #Time to train!

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    print (x_test.shape)

    #TODO: figure out where to initialise these parameters (Keras example + Beren)

    vae.fit(x_train, x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data= (x_test, x_test))

    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)

    if save:
        save_array((x_test, x_test_encoded), save_name + '_imgs_preds')

    """
    #Visualise the latent space? Not sure if this will work but here goes
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()

    #Display a 2D manifold of the digits - probably not at all relevant here

    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    #We will sample n points within [-15, 15] standard deviations
    grid_x = np.linspace(-15, 15, n)
    grid_y = np.linspace(-15, 15, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]]) * epsilon_std
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()
    """

def main():

    if len(sys.argv) >=2:
        fname = sys.argv[1]

    save_name = 'variational_1_epoch'

    run_variational(fname, save = True, save_name = save_name)

if __name__ == '__main__':
    main()
