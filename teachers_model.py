import argparse
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

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


class DataGenerator(tf.compat.v2.keras.utils.Sequence):

    def __init__(self, X_data, y_data, batch_size, in_dim, out_dim,
                 shuffle=True, validation=False,
                 train_size= 0.8, val_size=0.2):

        self.batch_size = batch_size
        self.X_data = X_data
        self.y_data = y_data
        self.n_classes = out_dim
        self.dim = in_dim
        self.shuffle = shuffle
        self.n = 0

        full_size = len(self.X_data)
        if not validation:
            train_size = self.get_size(train_size, full_size)
            self.indices = np.arange(train_size)
        else:
            val_size = full_size - self.get_size(val_size, full_size)
            self.indices = np.arange(val_size, full_size)

        self.on_epoch_end()

    def __next__(self):
        data = self.__getitem__(self.n)
        self.n += 1
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0
        return data

    def __len__(self):
        return math.ceil(len(self.indices)/self.batch_size)

    def __getitem__(self, index):
        bx = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X = self._generate_x(bx)
        if self.y_data is not None:
            y = self._generate_y(bx)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _generate_x(self, indices):
        X = np.empty((self.batch_size, *self.dim))
        for i, d in enumerate(indices):
            X[i] = self.X_data[d]
        return X

    def _generate_y(self, indices):
        y = np.empty((self.batch_size, self.n_classes))
        for i, d in enumerate(indices):
            y[i] = self.y_data[d]
        return y

    def get_size(self, x, size):
        if isinstance(x, float):
            return math.ceil(x*size)
        else:
            return x


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
    predicted = model.predict(X_test)

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
    parser.add_argument('-c', '--classes', type=int, default=5,
                        help='Number of classes', dest='n_classes')
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

    h5 = h5py.File(args.dataset_path, 'r')
    X_train = h5['X_train']
    y_train = h5['y_train']
    in_shape = (100, 100, 1)

    train_generator = DataGenerator(X_train, y_train,
                                    batch_size=args.batch_size,
                                    in_dim=in_shape,
                                    out_dim=args.n_classes,
                                    train_size=0.1,
                                    shuffle=True)

    val_generator = DataGenerator(X_train, y_train,
                                  batch_size=args.batch_size,
                                  in_dim=in_shape,
                                  out_dim=args.n_classes,
                                  shuffle=True,
                                  validation=True)
    model = build_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[CategoricalAccuracy(name="accuracy")])
    model.fit(train_generator,
              steps_per_epoch=len(train_generator),
              epochs=args.epochs,
              validation_data=val_generator,
              validation_steps=len(val_generator),
              callbacks=[ModelCheckpoint(args.save_path),
                         ReduceLROnPlateau(patience=args.patience)],
              use_multiprocessing=True,
              workers=args.workers)

    # Evaluate the model
    model = load_model(args.save_path)
    X_test = h5['X_test']
    y_test = h5['y_test']
    test_generator = DataGenerator(X_test, None,
                                   batch_size=100,
                                   in_dim=in_shape,
                                   out_dim=args.n_classes,
                                   shuffle=False,
                                   val_size=0.0)
    evaluate(model, test_generator, y_test[()])

    h5.close()
