import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Get data from cifar10

(x_train_full, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
x_train = x_train_full
#x_train = x_train_full[y_train[:,0]==1]
# Pre-proccess  data

x_train = x_train.astype(np.float32)/255.0
x_train = x_train * 2 -1


def disc():
    """
    Keres model, computes discriminator score
    
    Returns
    tensor with with batch_size score 
    """
    
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', input_shape=(32,32,3)))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    

    model.optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    
    return model

def gen(noise_size):
    """
    Keres model, generates image
    
    Iputs:
        tensor with batch_size noise
    Returns
        tensor with with batch_size images
    """

    model = keras.Sequential()
    model.add(keras.layers.Dense(4 * 4 * 512, input_shape=(noise_size,)))
    model.add(keras.layers.Reshape((4, 4, 512)))
    model.add(keras.layers.BatchNormalization(momentum=0.8))
    model.add(keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same', activation='relu')) # 8x8
    model.add(keras.layers.BatchNormalization(momentum=0.8))
    model.add(keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')) # 16x16
    model.add(keras.layers.BatchNormalization(momentum=0.8))
    model.add(keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, padding='same', activation='tanh')) # 32x32
             
    model.optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    
    return model


# Set hyper-parameters
num_epochs = 300
num_iterations = 500
batch_size = 100
noise_size = 100
K = 1

# Models

G = gen(noise_size)
D = disc()
images = []
disc_losses = []
gen_losses = []

# Training


for epoch in range(num_epochs):
    for batch in range(num_iterations):
        for k in range(K):
            with tf.GradientTape() as disc_tape:
                
                idx = random.randint(0, x_train.shape[0]-batch_size)
                real_data = x_train[idx:idx+batch_size]
                real_results = D(real_data)
                
                disc_noise = tf.random.uniform((batch_size, noise_size), minval=-1, maxval=1)
                disc_fake_images = G(disc_noise)
                fake_results = D(disc_fake_images)
                
                disc_loss = (keras.losses.binary_crossentropy(np.ones((batch_size,1)), real_results) + keras.losses.binary_crossentropy(np.zeros((batch_size,1)), fake_results)) / 2
                disc_grads = disc_tape.gradient(disc_loss, D.trainable_variables)
                
                D.optimizer.apply_gradients(zip(disc_grads, D.trainable_variables))
            
        with tf.GradientTape() as gen_tape:
            
            gen_noise = tf.random.uniform((batch_size, noise_size), minval=-1, maxval=1)
            gen_fake_images = G(gen_noise)
            gen_fake_results = D(gen_fake_images)
            
            gen_loss = keras.losses.binary_crossentropy(np.ones((batch_size, 1)), gen_fake_results)
            gen_grads = gen_tape.gradient(gen_loss, G.trainable_variables)
            
            G.optimizer.apply_gradients(zip(gen_grads, G.trainable_variables))
        if (batch % 50 == 0):
            print("Epoch: %d, Batch: %d, D loss: %s, G loss: %s" % (epoch, batch, np.mean(disc_loss.numpy()), np.mean(gen_loss.numpy())))
            noise = tf.random.uniform((1, noise_size), minval=0, maxval=1)
            img = G(noise)
            np_img = ((img.numpy().reshape(32,32,3) + 1)  * 128).astype(int)
            #plt.imshow(np_img)
            images.append(np_img)
            disc_losses.append(np.mean(disc_loss.numpy()))
            gen_losses.append(np.mean(gen_loss.numpy()))
            
                            
        
            