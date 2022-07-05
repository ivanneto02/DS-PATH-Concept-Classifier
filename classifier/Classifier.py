import tensorflow as tf
from tensorflow import keras

class TFIDF(tf.keras.Model):
    ''''This classifier will be trained on a corpus of diseases vs drugs documents.'''
    def __init__(self):
        super().__init__()

    def train(self, data):
        pass

    def test(self, data):
        pass