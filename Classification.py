import tensorflow as tf
import tensorflow_datasets as tfds


def createModel(info):
    encoder = info.features['text'].encoder
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(encoder.vocab_size, 64))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    return model


if __name__ == '__main__':
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                              as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    print(type(train_dataset))
