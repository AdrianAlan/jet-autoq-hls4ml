import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_io as tfio

from sklearn.metrics import roc_curve, auc
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D)


def build_model(input_shape=(100, 100, 1), drate=.25):

    in_image = Input(shape=(input_shape))

    x = Conv2D(14,
               kernel_size=6,
               data_format="channels_last",
               strides=(1, 1),
               padding="same")(in_image)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(drate)(x)
    x = Conv2D(6,
               kernel_size=4,
               data_format="channels_last",
               strides=(1, 1),
               padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(drate)(x)
    x = Flatten()(x)
    x = Dense(250, activation='relu')(x)
    output = Dense(5, activation='softmax')(x)

    return Model(inputs=in_image, outputs=output)


def evaluate(model, X_test, y_test, save_path='evaluation.png'):

    labels = ['gluon', 'quark', 'W', 'Z', 'top']

    predicted = model.predict(X_test, verbose=1)
    fig = plt.figure()
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_test[:, i], predicted[:, i])
        plt.plot(tpr, fpr,
                 label='%s tagger, auc = %.1f%%' % (label, auc(fpr, tpr)*100))
    plt.semilogy()
    plt.xlabel("sig. efficiency")
    plt.ylabel("bkg. mistag rate")
    plt.ylim(0.000001, 1)
    plt.grid(True)
    plt.legend(loc='lower right')

    fig.savefig(save_path)
    plt.close(fig)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Train the teacher's network")
    parser.add_argument('dataset_path', type=str, help='Path to dataset')
    parser.add_argument('save_path', type=str, help='Path to trained model')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='Number of training samples in a batch',
                        dest='batch_size')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='Number of training epochs', dest='epochs')
    parser.add_argument('-p', '--patience', type=int, default=2,
                        help='LR reduction callback patience',
                        dest='patience')
    parser.add_argument('-w', '--workers', type=int, default='4',
                        help='Number of workers', dest='workers')
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

    X_train = tfio.IODataset.from_hdf5(args.dataset_path, "/X_train")
    y_train = tfio.IODataset.from_hdf5(args.dataset_path, "/y_train")
    X_test = tfio.IODataset.from_hdf5(args.dataset_path, "/X_test")
    y_test = tfio.IODataset.from_hdf5(args.dataset_path, "/y_test")

    train_dataset = tf.data.Dataset.zip((X_train, y_train)) \
                                   .shuffle(10000).batch(args.batch_size) \
                                   .prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = tf.data.Dataset.zip((X_test, y_test)) \
                                  .batch(args.batch_size) \
                                  .prefetch(tf.data.experimental.AUTOTUNE)

    model = build_model()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[CategoricalAccuracy(name="accuracy")])

    model.fit(train_dataset,
              epochs=args.epochs,
              validation_data=test_dataset,
              callbacks=[ModelCheckpoint(args.save_path),
                         ReduceLROnPlateau(patience=args.patience)],
              use_multiprocessing=True,
              workers=args.workers)

    # Evaluate the model
    with h5py.File(args.dataset_path, 'r') as f:
        model = load_model(args.save_path)
        evaluate(model, test_dataset, f['y_test'][()])
