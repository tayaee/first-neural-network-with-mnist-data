import numpy as np
from tensorflow import keras


def get_new_samples(n=10):
    _, (x_test, y_test) = keras.datasets.mnist.load_data()
    indices = np.random.choice(len(x_test), n, replace=False)

    samples = []
    for idx in indices:
        raw_img = x_test[idx]
        flat_img = raw_img.reshape(-1).astype("float32") / 255
        samples.append({"image": raw_img, "flat": flat_img, "label": y_test[idx]})
    return samples
