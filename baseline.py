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
from keras import backend as K
from keras.callbacks import TensorBoard
from utils import *
from autoencoder import *
from experiments import *
import scipy
import sys

#DICE version 05/04
#No clue if this will work
def run_baseline(fname, save = False, save_name = None):

    input_img = Input(shape=(100, 100, 3))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    #Initially had binary crossentropy but we want to try MSE

    imgs = load(fname)
    imgs = np.array(imgs)
    x_train, x_test = split_into_test_train(imgs)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 100, 100, 3))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 100, 100, 3))  # adapt this if using `channels_first` image data format

    from keras.callbacks import TensorBoard

    autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))
    #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')] - final argument for when we are using DICE

    decoded_imgs = autoencoder.predict(x_test)

    if save:
        save_array((x_test, decoded_imgs), save_name + '_imgs_preds')


    """
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    """


#No split here so we can just use one hemisphere object
"""
def run_baseline_autoencoder(fname,epochs=100, save=True, test_up_to=None,
preview = False, verbose = True, param_name= None, param = None, save_name =
None, test_all=False):
    imgs = load(fname)
    imgs = np.array(imgs)
    imgs= normalise(imgs)
    train, test = split_into_test_train(imgs)
    if verbose:
        print (train.shape)
        print (test.shape)

    a1 = Hemisphere(train, train, test, test)

    a1.train(epochs=epochs)
    if verbose:
        print ("a1 trained")

    preds1, errmap1 = a1.get_error_maps(return_preds = True)
    if verbose:
        print (preds1.shape)
        print (errmap1.shape)


    if save:
        if save_name is None:
            save_array((redtest, preds1, errmap1),fname+'_imgs_preds_errmaps')
        if save_name is not None:
            save_array((redtest, preds1, errmap1), save_name + '_imgs_preds_errmaps')
"""

def main():

    """
    save_name = 'baseline_50_epochs'
    epochs = 50
    if len(sys.argv) >=2:
        fname = sys.argv[1]
    if len(sys.argv)>=3:
        save_name = sys.argv[2]
    if len(sys.argv)>=4:
        epochs = int(sys.argv[3])
    if len(sys.argv) <=1:
        raise ValueError('Need to input a filename for the data when running the model')


    run_baseline_autoencoder(fname,epochs= epochs, save=True, test_up_to=None, preview = False, verbose = True, param_name= None, param = None, save_name = save_name, test_all=False)
    """
    if len(sys.argv) >=2:
        fname = sys.argv[1]
    save_name = 'baseline_50_epochs'

    run_baseline(fname, save = True, save_name = save_name)

if __name__ == '__main__':
    main()
