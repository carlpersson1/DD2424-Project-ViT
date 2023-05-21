import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras import layers, models
import datetime
import os
import json

(in_train, out_train), (in_test, out_test) = cifar10.load_data()

# Convert integer labels to vectors with one-hot encoding

out_train = to_categorical(out_train, num_classes=10)
out_test = to_categorical(out_test, num_classes=10)

# Flatten images to vectors of length 32 * 32 * 3 == 3072

in_train = np.reshape(in_train,(in_train.shape[0], 3072))
in_test = np.reshape(in_test,(in_test.shape[0], 3072))
in_train = in_train.astype('float32')
in_test = in_test.astype('float32')

# Normalize values to be in range 0-1

in_train /= 255
in_test /= 255

# Build model layer by layer

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=3072))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Optional, set optizimer to stochastic gradient descent

# from tensorflow.keras.optimizers import SGD
# optimizer = SGD(learning_rate=0.01, weight_decay=1e-5, momentum=0.8)

# Compile model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model

history = model.fit(in_train, out_train, epochs=10, batch_size=32, verbose=1, validation_split=0.1)

# Get and print metrics

test_loss, test_acc = model.evaluate(in_test, out_test, verbose=0)

print('test_loss:', test_loss)
print('test_acc:', test_acc)

# Save training and validation history to file

cwd = os.getcwd()
data_file_name = f'mlp_loss_vs_epoch_data_{datetime.datetime.now()}.txt'
data_file_path = os.path.join(cwd, data_file_name)
with open(data_file_path, 'w') as data_file:
     data_file.write(json.dumps(history.history))

# Plot training and validation history

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs Epoch')
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
file_name = f'mlp_loss_vs_epoch_plot_{datetime.datetime.now()}.jpg'
file_path = os.path.join(cwd, file_name)
plt.savefig(file_path)
plt.show()

