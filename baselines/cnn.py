import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.datasets import cifar10
import datetime
import os
import json

(in_train, out_train), (in_test, out_test) = cifar10.load_data()

# Normalize values to be in range 0-1 and cast to float

in_train = in_train / 255.0
in_test = in_test / 255.0

# Build model layer by layer

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train model

history = model.fit(in_train, out_train, epochs=10, verbose=1,
                    validation_data=(in_test, out_test))

# Get and print metrics

test_loss, test_acc = model.evaluate(in_test, out_test, verbose=0)

print('test_loss:', test_loss)
print('test_acc:', test_acc)

# Save training and validation history to file

cwd = os.getcwd()
data_file_name = f'cnn_loss_vs_epoch_data_{datetime.datetime.now()}.txt'
data_file_path = os.path.join(cwd, data_file_name)
with open(data_file_path, 'w') as data_file:
     data_file.write(json.dumps(history.history))

# Plot training and validation history

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch number')
plt.ylabel('Loss')
# plt.ylim([0.5, 1])
plt.legend(['Training', 'Validation'], loc='upper right')
file_name = f'cnn_loss_vs_epoch_plot_{datetime.datetime.now()}.jpg'
file_path = os.path.join(cwd, file_name)
plt.savefig(file_path)
plt.show()


