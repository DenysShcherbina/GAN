import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import keras
import tensorflow as tf
from keras.layers import Dense, Input, Flatten, Reshape, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, LeakyReLU
import matplotlib.pyplot as plt
from keras.datasets import mnist
import keras._tf_keras.keras.backend as K
import time

# Data preparation
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[y_train == 7]
y_train = y_train[y_train == 7]

BUFFER_SIZE = x_train.shape[0]
BATCH_SIZE = 100
BUFFER_SIZE = BUFFER_SIZE // BATCH_SIZE * BATCH_SIZE
x_train = x_train[:BUFFER_SIZE]
y_train = y_train[:BUFFER_SIZE]

x_train = x_train / 255
x_test = x_test / 255
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# Create Neural Networks
hidden_dim = 2

# Create generator
generator = keras.Sequential([
    Input(shape=(hidden_dim,)),
    Dense(7*7*256, activation='relu'),
    BatchNormalization(),
    Reshape((7, 7, 256)),
    Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='sigmoid')])
# generator.summary()

# Create discriminator
discriminator = keras.Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
    LeakyReLU(),
    Dropout(0.3),
    Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
    LeakyReLU(),
    Dropout(0.3),
    Flatten(),
    Dense(1)])
# discriminator.summary()


# Loss function
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output):
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return loss


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optimizer = keras.optimizers.Adam(1e-4)


# Model training logic
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, hidden_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(dataset, epochs):
    history = []
    MAX_PRINT_LABEL = 10
    th = BUFFER_SIZE // (BATCH_SIZE * MAX_PRINT_LABEL)

    for epoch in range(1, epochs + 1):
        print(f'{epoch}/{epochs} :', end='')
        start = time.time()
        n = 0
        gen_loss_epoch = 0

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            gen_loss_epoch += K.mean(gen_loss)
            if (n // th == 0):
                print('=', end='')
            n += 1

        history += [gen_loss_epoch / n]
        print(': ' + str(history[-1]))
        print(f'Время эпохи {epoch} составляет {time.time() - start} секунд.')

    return history


# Start training models
EPOCHS = 100
history = train(train_dataset, EPOCHS)
plt.plot(history)
plt.grid(True)
plt.show()

# Visualization generation results
n = 2
total = 2 * n + 1

plt.figure(figsize=(total, total))
num = 1

for i in range(-n, n+1):
    for j in range(-n, n+1):
        ax = plt.subplot(total, total, num)
        num += 1

        img = generator.predict(np.expand_dims([0.5 * i/n, 0.5 * j/n], axis=0))
        plt.imshow(img[0, :, :, 0], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()

# Save our generator for future usage
keras.saving.save_model(model=generator, filepath='generator2.keras')
