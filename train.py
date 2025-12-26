import logging
import os

import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential

logger = logging.getLogger(__name__)


def train_and_save():
    (X_train, y_train), _ = keras.datasets.mnist.load_data()
    rows = X_train.shape[0]
    X_train = X_train.reshape(rows, -1)
    X_train = X_train.astype("float32")
    X_train = X_train / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    model = Sequential(
        [
            Dense(128, activation="relu", input_dim=784),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"],
    )
    early_stop = EarlyStopping(
        monitor="accuracy",
        min_delta=0.001,
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )
    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
    )

    if not os.path.exists("data"):
        os.makedirs("data")
    model.save("data/mnist_model.keras")
    print("Model saved to data/mnist_model.keras")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_and_save()
