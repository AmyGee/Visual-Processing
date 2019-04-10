from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.models import load_model
from keras.datasets import cifar10
from keras import optimizers
from utils import *
from keras.callbacks import TensorBoard
from autoencoder import *
from utils import *
from experiments import*

#Laptop version 09/04

#Following the tutorial at https://blog.keras.io/building-autoencoders-in-keras.html & including comments from gregorygunderson.com & jeremyjordan.me
#Eternal thanks to Beren Millidge for his gestalt_VAE code  (https://github.com/Bmillidgework/Generative-Gestalt-Autoencoders/blob/master/gestalt_vae.py) which helped me understand how to include convolutional layers


def vae_model(fname, save=True, save_name = None , verbose = True):

	imgs = load(fname)
	imgs = np.array(imgs)
	x_train, x_test = split_into_test_train(imgs)

	img_rows, img_cols, img_chns = 100, 100, 3
	if K.image_data_format() == 'channels_first':
			original_img_size = (img_chns, img_rows, img_cols)
	else:
			original_img_size = (img_rows, img_cols, img_chns)

	epochs = 50
	batch_size = 50
	# number of convolutional filters to use
	filters = 64
	# convolution kernel size
	num_conv = 3


	latent_dim = 2
	intermediate_dim = 128
	epsilon_std = 1.0
	activation = 'relu'
	# input image dimensions
	input_shape = (100, 100, 3) #Define shape without including the batch size

	#First we create the encoder network which maps inputs to our latent distribution parameters:
	x = Input(shape=input_shape)
	conv_1 = Conv2D(img_chns,
					kernel_size=(2, 2),
					padding='same', activation=activation)(x) #Inputs & outputs a 4D tensor
	if verbose:
		print (conv_1.shape)
	conv_2 = Conv2D(filters,
					kernel_size=(2, 2),
					padding='same', activation=activation,
					strides=(2, 2))(conv_1)
	if verbose:
		print (conv_2.shape)
	conv_3 = Conv2D(filters,
					kernel_size=num_conv,
					padding='same', activation=activation,
					strides=1)(conv_2)
	if verbose:
		print (conv_3.shape)
	conv_4 = Conv2D(filters,
					kernel_size=num_conv,
					padding='same', activation=activation,
					strides=2)(conv_3)
	if verbose:
		print (conv_4.shape)
	flat = Flatten()(conv_4) #For generating the latent vector
	if verbose:
		print (flat.shape)
	hidden = Dense(intermediate_dim, activation=activation)(flat)
	if verbose:
		print (hidden.shape)

	z_mean = Dense(latent_dim)(hidden)
	z_log_sigma = Dense(latent_dim)(hidden) #Use log variance instead of standard deviation as it is more convenient and helps with numerical stability

	#Use the latent distribution parameters to sample new & similar points from the latent space
	def sampling(args):
		z_mean, z_log_sigma = args
		epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
								  mean=0., stddev=epsilon_std) #Normally distributed values in a tensor to be used as noise
		return z_mean + K.exp(z_log_sigma) * epsilon
	#Shifting the random sample by the mean and scaling it by the variance

	#Make the sampling the input
	z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma]) #Function, output shape multiplied by args
	#Lambda wraps an arbitrary expression as a layer, so here we are wrapping the sampling (latent space) as our input layer

	#Map the latent points to reconstructed inputs
	decoder_hid = Dense(intermediate_dim, activation=activation)

	decoder_upsample = Dense(filters * int(img_rows/4) * int(img_cols/4), activation=activation)

	if K.image_data_format() == 'channels_first':
		output_shape = (batch_size, filters, int(img_rows/4), int(img_cols/4))
	else:
		output_shape = (batch_size, int(img_rows/4), int(img_cols/4), filters)

	decoder_reshape = Reshape(output_shape[1:]) #Reshapes the output
	decoder_deconv_1 = Conv2DTranspose(filters,
									   kernel_size=num_conv,
									   padding='same',
									   strides=1,
									   activation=activation) #Transposed layers for deconvolution
	decoder_deconv_2 = Conv2DTranspose(filters,
									   kernel_size=num_conv,
									   padding='same',
									   strides=2,
									   activation=activation)
	if K.image_data_format() == 'channels_first':
		output_shape = (batch_size, filters, int(img_rows+1), int(img_cols+1))
	else:
		output_shape = (batch_size, int(img_rows+1), int(img_cols+1), filters)
	decoder_deconv_3_upsamp = Conv2DTranspose(filters,
											  kernel_size=(3, 3),
											  strides=(2, 2),
											  padding='valid',
											  activation=activation)
	decoder_mean_squash = Conv2D(img_chns,
								 kernel_size=2,
								 padding='valid',
								 activation='sigmoid')

	hid_decoded = decoder_hid(z)
	up_decoded = decoder_upsample(hid_decoded)
	reshape_decoded = decoder_reshape(up_decoded)
	deconv_1_decoded = decoder_deconv_1(reshape_decoded)
	deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
	x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
	x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)


	#Instantiate 3 models

	#End-to-end autoencoder for mapping inputs to reconstructions
	vae = Model(x, x_decoded_mean_squash)

	kl = kl_loss(z_mean, z_log_sigma)
	vae.add_loss(kl)

	#Encoder mapping from inputs to latent space
	encoder = Model(x, z_mean)

	#Generator which takes points from the latent space to output the reconstructed samples
	decoder_input = Input(shape=(latent_dim,))
	_hid_decoded = decoder_hid(decoder_input)
	_up_decoded = decoder_upsample(_hid_decoded)
	_reshape_decoded = decoder_reshape(_up_decoded)
	_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
	_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
	_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
	_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
	#Push z through decoder
	generator = Model(decoder_input, _x_decoded_mean_squash)


	#Time to train!

	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	print (x_test.shape)

	shape = x_train.shape[1:]

	#Train using the end-to-end model with a custom loss & K-L divergence regularization
	"""def vae_loss(x, x_decoded_mean):
		xent_loss = metrics.binary_crossentropy(x, x_decoded_mean_squash) #Reconstruction loss
		#Binary crossentropy because the decoding term is a Bernoulli multi layered perceptron - is it worth also trying Gaussian + MSE??
		kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1) #Variational loss
		return xent_loss + kl_loss #Combine the losses
	"""
	#Can only get these losses to work when we define them outside of the VAE model
	vae.compile(optimizer='adam',loss= reconstruction_loss)
	vae.summary()

	vae.fit(x_train, x_train,
			shuffle=True,
			epochs=epochs,
			batch_size=batch_size,
			validation_data= (x_test, x_test))

	predictions = vae.predict(x_test, batch_size=batch_size)

	if save:
		save_array((x_test, predictions), save_name + '_imgs_preds')

#Need to define losses here because it wasn't working trying to define them in the vae_model definition
def reconstruction_loss(y, x_decoded):
	#let's hard code this for now
	rows = 8
	cols = 32
	rec_loss = rows * cols * metrics.binary_crossentropy(K.flatten(y), K.flatten(x_decoded))
	print("Rec loss: " + str(rec_loss))
	return rec_loss

def kl_loss(z_mean, z_log_sigma):
	klloss =  -0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
	#print("KL loss: " + str(klloss))
	return klloss

def main():

	if len(sys.argv) >=2:
		fname = sys.argv[1]

	save_name = 'variational_50_epochs'

	vae_model(fname, save = True, save_name = save_name)

if __name__ == '__main__':
	main()
