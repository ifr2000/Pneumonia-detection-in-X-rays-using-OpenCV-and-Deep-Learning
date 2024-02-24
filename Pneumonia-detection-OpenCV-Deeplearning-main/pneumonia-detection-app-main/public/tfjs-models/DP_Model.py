from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam

# Define the model architecture
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3), padding='same', name='conv2d_5'), # Convolutional layer with 16 filters, relu activation
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='max_pooling2d_5'), # Max pooling layer
    Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_6'), # Convolutional layer with 32 filters, relu activation
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='max_pooling2d_6'), # Max pooling layer
    Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_7'), # Convolutional layer with 64 filters, relu activation
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='max_pooling2d_7'), # Max pooling layer
    Conv2D(96, (3, 3), activation='relu', padding='same', name='conv2d_8'), # Convolutional layer with 96 filters, relu activation
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='max_pooling2d_8'), # Max pooling layer
    Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_9'), # Convolutional layer with 128 filters, relu activation
    BatchNormalization(axis=3, momentum=0.99, epsilon=0.001, name='batch_normalization_1'), # Batch normalization layer
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='max_pooling2d_9'), # Max pooling layer
    Flatten(name='flatten_1'), # Flatten layer
    Dense(256, activation='relu', name='dense_3'), # Fully connected layer with 256 units, relu activation
    Dropout(0.1, name='dropout_2'), # Dropout layer
    Dense(128, activation='relu', name='dense_4'), # Fully connected layer with 128 units, relu activation
    Dropout(0.1, name='dropout_3'), # Dropout layer
    Dense(1, activation='sigmoid', name='dense_5') # Output layer with sigmoid activation
])

# Compile the model
optimizer = Adam(learning_rate=2.429999995001708e-06, beta_1=0.8999999761581421, beta_2=0.9990000128746033, epsilon=1e-07, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Load the weights
weights_manifest = [
    {"name": "batch_normalization_1/gamma", "shape": [128], "dtype": "float32"},
    {"name": "batch_normalization_1/beta", "shape": [128], "dtype": "float32"},
    {"name": "batch_normalization_1/moving_mean", "shape": [128], "dtype": "float32"},
    {"name": "batch_normalization_1/moving_variance", "shape": [128], "dtype": "float32"},
    {"name": "conv2d_5/kernel", "shape": [3, 3, 3, 16], "dtype": "float32"},
    {"name": "conv2d_5/bias", "shape": [16], "dtype": "float32"},
    {"name": "conv2d_6/kernel", "shape": [3, 3, 16, 32], "dtype": "float32"},
    {"name": "conv2d_6/bias", "shape": [32], "dtype": "float32"},
    {"name": "conv2d_7/kernel", "shape": [3, 3, 32, 64], "dtype": "float32"},
    {"name": "conv2d_7/bias", "shape": [64], "dtype": "float32"},
    {"name": "conv2d_8/kernel", "shape": [3, 3, 64, 96], "dtype": "float32"},
    {"name": "conv2d_8/bias", "shape": [96], "dtype": "float32"},
    {"name": "conv2d_9/kernel", "shape": [3, 3, 96, 128], "dtype": "float32"},
    {"name": "conv2d_9/bias", "shape": [128], "dtype": "float32"},
    {"name": "dense_3/kernel", "shape": [512, 256], "dtype": "float32"},
    {"name": "dense_3/bias", "shape": [256], "dtype": "float32"},
    {"name": "dense_4/kernel", "shape": [256, 128], "dtype": "float32"},
    {"name": "dense_4/bias", "shape": [128], "dtype": "float32"},
    {"name": "dense_5/kernel", "shape": [128, 1], "dtype": "float32"},
    {"name": "dense_5/bias", "shape": [1], "dtype": "float32"}
]

for layer in weights_manifest:
    weights = [np.fromfile('group1-shard1of1.bin', dtype=layer['dtype'])] # Load weights from file
    model.get_layer(layer['name']).set_weights(weights) # Set weights for each layer

# Summary of the model
model.summary()