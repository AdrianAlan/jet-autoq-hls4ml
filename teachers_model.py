import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

from kerastuner import Hyperband
from kerastuner import HyperModel
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    ReLU)
from sklearn.metrics import roc_curve, auc


class JetHyperModel(HyperModel):

    def __init__(self, input_shape=(100, 100, 1), output_size=5):
        self.input_shape = input_shape
        self.output_size = output_size

    def build(self, hp):

        in_image = Input(shape=(self.input_shape))

        x = in_image
        for layer_id in range(hp.Int('conv_layers', 1, 4, default=2)):
            layer_id = str(layer_id)
            x = Conv2D(hp.Int('filters_' + layer_id, 6, 18, step=4),
                       kernel_size=hp.Int('kernel_size' + layer_id, 2, 5),
                       data_format="channels_last",
                       strides=(1, 1),
                       padding="same")(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            if hp.Choice('pooling' + layer_id, ['max', 'avg']) == 'max':
                x = MaxPooling2D(pool_size=(2, 2))(x)
            else:
                x = AveragePooling2D(pool_size=(2, 2))(x)

            x = Dropout(hp.Choice('droupout_rate', [0., 0.2, 0.5]))(x)

        x = Flatten()(x)

        for layer_id in range(hp.Int('dense_layers', 0, 5, default=1)):
            layer_id = str(layer_id)
            x = Dense(hp.Int('units_' + layer_id, 5, 50, step=5))(x)
            x = ReLU()(x)
            x = Dropout(hp.Choice('droupout_rate', [0., 0.2, 0.5]))(x)

        output = Dense(self.output_size, activation='softmax')(x)

        model = Model(inputs=in_image, outputs=output)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=[CategoricalAccuracy(name="accuracy")])

        return model


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
    parser.add_argument('project_name', type=str, help='Project name')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='Number of training samples in a batch',
                        dest='batch_size')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='Number of training epochs', dest='epochs')
    parser.add_argument('-s', '--steps', type=int, default=1000,
                        help='Number of steps per epoch', dest='steps')
    parser.add_argument('-p', '--patience', type=int, default=2,
                        help='LR reduction callback patience',
                        dest='patience')
    parser.add_argument('-w', '--workers', type=int, default='4',
                        help='Number of workers', dest='workers')
    args = parser.parse_args()

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

    # Search for hyperparameters
    hypermodel = JetHyperModel()

    tuner = Hyperband(hypermodel,
                      objective='val_accuracy',
                      executions_per_trial=3,
                      max_epochs=args.epochs,
                      directory=args.save_path,
                      project_name=args.project_name)

    tuner.search(train_dataset,
                 epochs=args.epochs,
                 steps_per_epoch=args.steps,
                 validation_data=test_dataset,
                 validation_steps=args.steps,
                 callbacks=[EarlyStopping('val_accuracy',
                                          patience=args.patience)],
                 use_multiprocessing=True,
                 workers=args.workers)

    # Retrain the best model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    save_dst = args.save_path + '/best.h5'
    model = tuner.hypermodel.build(best_hps)
    model.fit(train_dataset,
              epochs=10*args.epochs,
              validation_data=test_dataset,
              callbacks=[EarlyStopping('val_accuracy',
                         patience=10*args.patience),
                         ModelCheckpoint(save_dst),
                         ReduceLROnPlateau(patience=args.patience)],
              use_multiprocessing=True,
              workers=args.workers)

    # Evaluate the model
    with h5py.File(args.dataset_path, 'r') as f:
        model = load_model(save_dst)
        evaluate(model, test_dataset, f['y_test'][()])
