import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from IPython import display
import matplotlib.pyplot as plt
import time

#x_train = put your dataset in here, you can use ImageDataGenerator or image_dataset_from_directory to get the data folder and rescale them, the dataset i used called (img_align_celeba) dataset it contained picture of celebraties

EPOCHS = 50
BATCH_SIZE = 256
BUFFER_SIZE = 60000
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def build_generator():
    gen = tf.keras.models.Sequential()
    gen.add(layers.Dense(8*8*256, use_bias=False, input_shape=(NOISE_DIM,)))
    gen.add(layers.BatchNormalization())
    gen.add(layers.LeakyReLU())

    gen.add(layers.Reshape((8, 8, 256)))
    assert gen.output_shape == (None, 8, 8, 256)

    gen.add(layers.Conv2DTranspose(128, 5, strides=2, padding='same', use_bias=False))
    assert gen.output_shape == (None, 16, 16, 128)
    gen.add(layers.BatchNormalization())
    gen.add(layers.LeakyReLU())

    gen.add(layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False))
    assert gen.output_shape == (None, 32, 32, 64)
    gen.add(layers.BatchNormalization())
    gen.add(layers.LeakyReLU())

    gen.add(layers.Conv2DTranspose(3, 5, strides=4, padding='same', use_bias=False, activation='tanh'))
    assert gen.output_shape == (None, 128, 128, 3)

    return gen

def build_discriminator():
    dis = tf.keras.models.Sequential()
    dis.add(layers.Conv2D(64, 5, strides = 2, padding = 'same',
                          input_shape = [28, 28, 1]))
    dis.add(layers.LeakyReLU())
    dis.add(layers.Dropout(0.3))

    dis.add(layers.Conv2D(128, 5, strides = 2, padding = 'same',))
    dis.add(layers.LeakyReLU())
    dis.add(layers.Dropout(0.3))

    dis.add(layers.Flatten())
    dis.add(layers.Dense(1))

    return dis

generator = build_generator()
discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(3e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(3e-4)
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

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

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)
def generate_and_save_images(model, epoch, test_input):
   
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

train(train_dataset, EPOCHS)