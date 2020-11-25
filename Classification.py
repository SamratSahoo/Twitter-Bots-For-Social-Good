import tensorflow as tf
import os
import tensorflow_hub as hub
from TextProcess import cleanText
import pandas as pd


class DepressionClassifier():
    def __init__(self, loadMode=False):
        # Data + Labels
        self.trainData, self.trainLabels = self.loadData('trainData.csv')  # 15,000 Samples (Each)
        self.testData, self.testLabels = self.loadData('testData.csv')  # 15,000 Samples (Each)

        # Pretrained Network
        self.embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
        self.hubLayer = hub.KerasLayer(self.embedding, input_shape=[],
                                       dtype=tf.string, trainable=True)
        self.modelName = 'model.h5'
        if not loadMode:
            # Actual Model built with Keras
            self.model = self.initModel()
            self.saveModel()
        else:
            self.model = self.loadModel()

        # self.model.summary()

    def loadData(self, fileName):
        data = pd.read_csv('Data' + os.sep + fileName)
        data.fillna(value='', inplace=True)
        data = data.sample(frac=1)
        return data._get_column_array(0), data._get_column_array(1)

    def initModel(self):
        model = tf.keras.Sequential()
        model.add(self.hubLayer)
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(16))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(self.trainData, self.trainLabels,
                            epochs=5,
                            validation_split=0.2,
                            verbose=1)
        return model

    def evaluateModel(self):
        results = self.model.evaluate(self.testData, self.testLabels, verbose=1)

    def predictDepression(self, text):
        text = cleanText(text)
        text = tf.expand_dims(tf.convert_to_tensor(text), axis=0)
        return self.model.predict(text)[0][0] > 1.0

    def saveModel(self):
        self.model.save('Models' + os.sep + self.modelName)

    def loadModel(self):
        return tf.keras.models.load_model('Models' + os.sep + self.modelName,
                                          custom_objects={'KerasLayer': hub.KerasLayer})


if __name__ == '__main__':
    # Step 1: Convert All words to numbers (Lookup table??)
    # Step 2: Convert numbers to multidimensional embeddings
    # Step 3: Feed through Neural Net
    classifer = DepressionClassifier(loadMode=True)
    print(classifer.predictDepression("To get close to another you have to take off your armor. It takes more  to be soft than tough.  reduces"))
    classifer.evaluateModel()
