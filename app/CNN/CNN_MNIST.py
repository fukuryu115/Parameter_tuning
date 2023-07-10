import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import os

# データセットのディレクトリパス
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "../dataset")

# 画像データの前処理
datagen = ImageDataGenerator(rescale=1./255)

# 訓練データの読み込みと前処理
train_generator = datagen.flow_from_directory(
    os.path.join(dataset_dir, "train"),
    target_size=(28, 28),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=32
)

# テストデータの読み込みと前処理
test_generator = datagen.flow_from_directory(
    os.path.join(dataset_dir, "test"),
    target_size=(28, 28),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=32
)

# CNNモデルの構築
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# TensorBoardコールバックの設定
log_dir = os.path.join(current_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, port=6001)

# モデルの訓練
model.fit(train_generator,
          steps_per_epoch=train_generator.n // train_generator.batch_size,
          epochs=5,
          validation_data=test_generator,
          validation_steps=test_generator.n // test_generator.batch_size,
          callbacks=[tensorboard_callback])
