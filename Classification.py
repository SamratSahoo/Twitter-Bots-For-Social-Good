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
        # Model name
        self.modelName = 'model.h5'

        # Use not saved model
        if not loadMode:
            # Actual Model built with Keras
            self.model = self.initModel()
            self.saveModel()
        else:
            # Use saved model
            self.model = self.loadModel()

        # Print model details
        self.model.summary()

    def loadData(self, fileName):
        # Grab Data
        data = pd.read_csv('Data' + os.sep + fileName)
        # Fix dataframe
        data.fillna(value='', inplace=True)
        data = data.sample(frac=1)

        # Returns Tweet + Label
        return data._get_column_array(0), data._get_column_array(1)

    def initModel(self):
        # Uses Keras Sequential Model
        model = tf.keras.Sequential()

        # Hub Layer = Pretrained Model
        model.add(self.hubLayer)

        # Fine tune model
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(256))

        # Dropout to avoid overfitting
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(16))
        model.add(tf.keras.layers.Dense(1))

        # Compile with Binary Cross Entropy
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(self.trainData, self.trainLabels,
                            epochs=5,
                            validation_split=0.2,
                            verbose=1)

        # Return Model
        return model

    def evaluateModel(self):
        # Print results based on test data
        results = self.model.evaluate(self.testData, self.testLabels, verbose=1)

    # Predict depression based on text
    def predictDepression(self, text):
        # clean text
        text = cleanText(text)
        # Fix the vector dimension problems
        text = tf.expand_dims(tf.convert_to_tensor(text), axis=0)
        #  > -1.0 I found was a safer threshold than > 0
        return self.model.predict(text)[0][0] > -2.0

    # Save model to file
    def saveModel(self):
        self.model.save('Models' + os.sep + self.modelName)

    # Load model from file
    def loadModel(self):
        return tf.keras.models.load_model('Models' + os.sep + self.modelName,
                                          custom_objects={'KerasLayer': hub.KerasLayer})


if __name__ == '__main__':
    # Initialize Classifier
    classifier = DepressionClassifier(loadMode=True)
    print(classifier.predictDepression("This world is such a sad and depressing world to live in T_T"))
    classifier.evaluateModel()
