import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Get data from cifar10

(x_train_full, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
#x_train = x_train_full
x_train = x_train_full[y_train[:,0]==1]
# Pre-proccess  data

x_train = x_train.astype(np.float32)/255.0
x_train = x_train * 2 -1

# Set hyper-parameters
num_epochs = 100
num_iterations = 500
batch_size = 64
noise_size = 100
alpha = 0.00005
c = 0.01
ncritic = 5
K = 1


def critic():
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
    model.add(keras.layers.Dense(1, activation='linear'))
    

    model.optimizer = keras.optimizers.RMSprop(learning_rate=alpha)
    
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
             
    model.optimizer = keras.optimizers.RMSprop(learning_rate=alpha)
    
    return model




# Models

G = gen(noise_size)
C = critic()
images = []
crit_losses_real = []
crit_losses_fake = []
gen_losses = []

# Training


for epoch in range(num_epochs):
    for batch in range(num_iterations):
        for n in range(ncritic):
            with tf.GradientTape() as disc_tape:
                
                idx = random.randint(0, x_train.shape[0]-batch_size)
                real_data = x_train[idx:idx+batch_size]
                real_results = C(real_data)
                
                disc_noise = tf.random.uniform((batch_size, noise_size), minval=-1, maxval=1)
                disc_fake_images = G(disc_noise)
                fake_results = C(disc_fake_images)
                
                disc_loss = tf.math.reduce_mean(real_results) - tf.math.reduce_mean(fake_results)
                disc_grads = disc_tape.gradient(disc_loss, C.trainable_variables)
                clipped_disc_grads = [tf.clip_by_value(grad, -c, c) for grad in disc_grads]
                
                C.optimizer.apply_gradients(zip(clipped_disc_grads, C.trainable_variables))
            crit_losses_real.append(tf.math.reduce_mean(real_results).numpy())
            crit_losses_fake.append(tf.math.reduce_mean(fake_results).numpy())
        with tf.GradientTape() as gen_tape:
            
            gen_noise = tf.random.uniform((batch_size, noise_size), minval=-1, maxval=1)
            gen_fake_images = G(gen_noise)
            gen_fake_results = C(gen_fake_images)
            
            gen_loss = tf.math.reduce_mean(gen_fake_results)
            gen_grads = gen_tape.gradient(gen_loss, G.trainable_variables)
            
            G.optimizer.apply_gradients(zip(gen_grads, G.trainable_variables))
            gen_losses.append(gen_loss.numpy())
        if (batch % 50 == 0):
            print("Epoch: %d, Batch: %d, C loss: %s, G loss: %s" % (epoch, batch, np.mean(disc_loss.numpy()), np.mean(gen_loss.numpy())))
            noise = tf.random.uniform((1, noise_size), minval=0, maxval=1)
            img = G(noise)
            np_img = ((img.numpy().reshape(32,32,3) + 1)  * 128).astype(int)
            #plt.imshow(np_img)
            images.append(np_img)
            #disc_losses.append(disc_loss.numpy())
            #gen_losses.append(gen_loss.numpy())
            
                            
        
            