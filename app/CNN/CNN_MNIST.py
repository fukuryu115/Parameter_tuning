import tensorflow as tf
from tensorflow.keras import layers, models
import os

# datasetディレクトリのパス
dataset_path = "dataset/"

# MNISTデータセットをロードして準備する
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# データの正規化
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0

# CNNモデルの構築
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルの訓練
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# モデルの評価
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
