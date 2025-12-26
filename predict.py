import numpy as np
import tensorflow as tf


class MNISTPredictor:
    def __init__(self, model_path="data/mnist_model.keras"):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image_flat):
        # image_flat shape: (784,)
        prediction = self.model.predict(image_flat.reshape(1, 784), verbose=0)
        return np.argmax(prediction), np.max(prediction)
