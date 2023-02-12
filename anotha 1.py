import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_train = (X_train - 127.5) / 127.5

# Set up the training data generator
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Use tf.data to create a batch generator
train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Build the generator model
def build_generator():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.LeakyReLU())

  model.add(tf.keras.layers.Reshape((7, 7, 256)))
  assert model.output_shape == (None, 7, 7, 256)

  model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
  assert model.output_shape == (None, 7, 7, 128)
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.LeakyReLU())

  model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 14, 14, 64)
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.LeakyReLU())

  model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
  assert model.output_shape == (None, 28, 28, 1)

  return model

# Build the discriminator model
def build_discriminator():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Dropout(0.3))
  
  model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Dropout(0.3))

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

  return model

# Build the GAN model
def build_gan(generator, discriminator):
  model = tf.keras.Sequential()
  model.add(generator)
  model.add(discriminator)
  return model

# Instantiate the generator and discriminator models
generator = build_generator()
discriminator = build_discriminator()

# Compile the discriminator model
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Freeze the weights of the discriminator model
discriminator.trainable = False

# Compile the GAN model
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Train the GAN model
def train(gan, discriminator, epochs=100, batch_size=256):
  # Store the losses for plotting
  losses = []
  
  # Train for the specified number of epochs
  for epoch in range(epochs):
    # Get a batch of real images
    images = next(iter(train_dataset))
    
    # Generate a batch of fake images
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_images = generator.predict(noise)
    
    # Train the discriminator model on real and fake images
    X = np.concatenate([images, fake_images])
    y = np.ones([2 * batch_size, 1])
    y[batch_size:, :] = 0
    d_loss, d_acc = discriminator.train_on_batch(X, y)
    
    # Generate more noise to train the generator model
    noise = np.random.normal(0, 1, (batch_size, 100))
    
    # Train the generator model
    g_loss = gan.train_on_batch(noise, np.ones([batch_size, 1]))
    
    # Store the loss values
    losses.append((d_loss, g_loss))
    
    # Print the progress
    if (epoch + 1) % 10 == 0:
      print(f'Epoch {epoch + 1}/{epochs}, Discriminator Loss: {d_loss:.4f}, Generator Loss: {g_loss:.4f}, Discriminator Accuracy: {d_acc:.4f}')
      
  return losses

# Train the GAN
losses = train(gan, discriminator, epochs=100, batch_size=256)

# Plot the losses
d_losses, g_losses = losses[:2]

g_losses = [x[1] for x in losses]
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.legend()
plt.show()

# Plot some generated images
noise = np.random.normal(0, 1, (25, 100))
generated_images = generator.predict(noise)
generated_images = generated_images.reshape(-1, 28, 28)
plt.figure(figsize=(10, 10))
for i in range(25):
  plt.subplot(5, 5, i + 1)
  plt.imshow(generated_images[i], cmap='gray')
  plt.axis('off')
plt.show()