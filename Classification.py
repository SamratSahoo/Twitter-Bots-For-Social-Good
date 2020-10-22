import tensorflow as tf
import os
import tensorflow_hub as hub
import numpy as np


class DepressionClassifier():
    def __init__(self):
        self.trainData = self.loadData('Data' + os.sep + 'trainData.csv')  # 15,000 Samples (Each)
        self.testData = self.loadData('Data' + os.sep + 'testData.csv')  # 25,000 Samples (Each)
        self.validationData = self.loadData('Data' + os.sep + 'validationData.csv')  # 10,000 (Each)
        self.embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
        self.hubLayer = hub.KerasLayer(self.embedding, input_shape=[],
                                       dtype=tf.string, trainable=True)

        self.model = self.initModel()
        self.model.summary()

    def loadData(self, fileName):
        return tf.data.experimental.make_csv_dataset(file_pattern=fileName, batch_size=32)

    def initModel(self):
        model = tf.keras.Sequential()
        model.add(self.hubLayer)
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(8))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(self.trainData.shuffle(10000).batch(512),
                            epochs=10,
                            validation_data=self.validationData.batch(512),
                            verbose=1)
        return model

    def evaluateModel(self):
        results = self.model.evaluate(self.testData.batch(512), verbose=1)

    def predict(self, text):
        pass

    def saveModel(self):
        self.model.save('Models' + os.sep + 'depressionClassifier')


if __name__ == '__main__':
    # Step 1: Convert All words to numbers (Lookup table??)
    # Step 2: Convert numbers to multidimensional embeddings
    # Step 3: Feed through Neural Net
    pass
