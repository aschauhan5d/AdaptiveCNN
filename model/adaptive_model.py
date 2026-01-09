
import tensorflow as tf
from tensorflow.keras import layers, models

class AdaptiveCNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_blocks = []

    def add_conv_block(self):
        self.conv_blocks.append(
            layers.Conv2D(32, (3,3), activation='relu', padding='same')
        )

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=self.input_shape))

        for conv in self.conv_blocks:
            model.add(conv)
            model.add(layers.MaxPooling2D())

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
