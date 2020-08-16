import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

from kerastuner import HyperModel, BayesianOptimization
from qkeras.autoqkeras import *
from qkeras.utils import *
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau)
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    DepthwiseConv2D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D)
from sklearn.metrics import roc_curve, auc


class JetHyperModel(HyperModel):

    def __init__(self, metrics, input_shape=(100, 100, 1), output_size=5):
        self.input_shape = input_shape
        self.output_size = output_size
        self.metrics = metrics

    def build(self, hp):

        in_image = Input(shape=(self.input_shape), name='input')

        x = in_image
        for layer_id in range(hp.Int('conv_layers', 1, 4, default=2)):
            d = hp.Choice('droupout_rate', [0., 0.2, 0.5])
            f = hp.Int('filters_%s' % layer_id, 4, 64, step=4)
            ks = hp.Int('kernel_size_%s' % layer_id, 2, 5)

            x = DepthwiseConv2D(ks, data_format="channels_last",
                                use_bias=False,
                                name=('dwconv_C%s' % layer_id))(x)
            x = Conv2D(f, (1, 1), data_format="channels_last",
                       name=('1x1conv_C%s' % layer_id))(x)
            x = BatchNormalization(name=('batch_norm_C%s' % layer_id))(x)
            x = Activation(activation='relu', name=('relu_C%s' % layer_id))(x)

            if hp.Choice(('pooling_%s' % layer_id), ['max', 'avg']) == 'max':
                x = MaxPooling2D((2, 2), name=('mpool_C%s' % layer_id))(x)
            else:
                x = AveragePooling2D((2, 2), name=('apool_C%s' % layer_id))(x)

            x = Dropout(d, name=('droupout_C%s' % layer_id))(x)

        x = Flatten()(x)

        for layer_id in range(hp.Int('dense_layers', 0, 5, default=1)):
            d = hp.Choice('dense_droupout_rate', [0., 0.2, 0.5])
            u = hp.Int('units_%s' % layer_id, 5, 50, step=5)

            x = Dense(u, name='dense_D%s' % layer_id)(x)
            x = BatchNormalization(name=('batch_norm_D%s' % layer_id))(x)
            x = Activation(activation='relu', name=('relu_D%s' % layer_id))(x)
            x = Dropout(d, name='dropout_D%s' % layer_id)(x)

        output = Dense(self.output_size,
                       activation='softmax',
                       name='output_softmax')(x)

        model = Model(inputs=in_image, outputs=output)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=self.metrics)

        return model


def evaluate(model, X_test, y_test):

    fig = plt.figure()
    labels = ['gluon', 'quark', 'W', 'Z', 'top']
    predicted = model.predict(X_test, verbose=1)
    for i, l in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_test[:, i], predicted[:, i])
        plt.plot(tpr, fpr,
                 label='{0} tagger, AUC={1:.1f}'.format(l, auc(fpr, tpr)*100))
    plt.xlabel("sig. efficiency")
    plt.ylabel("bkg. mistag rate")
    plt.ylim(0.000001, 1)
    plt.semilogy()
    plt.grid(True)
    plt.legend(loc='lower right')
    fig.savefig(save_path)
    plt.close(fig)


def print_model_to_json(keras_model, outfile_name):
    with open(outfile_name, 'w') as f:
        json.dump(json.loads(keras_model.to_json()), f,
                  sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Train the teacher's network")
    parser.add_argument('dataset_path', type=str, help='Path to dataset')
    parser.add_argument('save_path', type=str, help='Path to trained model')
    parser.add_argument('-b', '--batch-size', type=int, default=100,
                        help='Number of training samples in a batch',
                        dest='batch_size')
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='Number of training epochs', dest='epochs')
    parser.add_argument('-s', '--steps', type=int, default=1000,
                        help='Number of steps per epoch', dest='steps')
    parser.add_argument('-t', '--trials', type=int, default=1000,
                        help='Number of AutoQ trials', dest='trials')
    parser.add_argument('-p', '--patience', type=int, default=2,
                        help='LR reduction callback patience',
                        dest='patience')
    parser.add_argument('-w', '--workers', type=int, default='10',
                        help='Number of workers', dest='workers')
    parser.add_argument('-q', '--autoq-config', type=str, dest='autoq_config',
                        help='Path tp AutoQKeras configuration')
    args = parser.parse_args()

    # Supress tf logger
    tf.get_logger().setLevel('ERROR')

    # Prepare the dataset
    X_train = tfio.IODataset.from_hdf5(args.dataset_path, "/X_train")
    y_train = tfio.IODataset.from_hdf5(args.dataset_path, "/y_train")
    X_test = tfio.IODataset.from_hdf5(args.dataset_path, "/X_test")
    y_test = tfio.IODataset.from_hdf5(args.dataset_path, "/y_test")

    train_dataset = tf.data.Dataset.zip((X_train, y_train)) \
                                   .shuffle(10000).batch(args.batch_size) \
                                   .prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = tf.data.Dataset.zip((X_test, y_test)) \
                                  .shuffle(10000).batch(args.batch_size) \
                                  .prefetch(tf.data.experimental.AUTOTUNE)

    # Set evaluation metrics
    metrics = [AUC(name="accuracy")]

    # Search for hyperparameters
    hypermodel = JetHyperModel(metrics=metrics)

    tuner = BayesianOptimization(hypermodel,
                                 objective='val_accuracy',
                                 max_trials=args.trials,
                                 directory=args.save_path,
                                 project_name='hypermodel')

    tuner.search(train_dataset,
                 epochs=args.epochs,
                 steps_per_epoch=args.steps,
                 validation_data=test_dataset,
                 validation_steps=args.steps,
                 callbacks=[EarlyStopping('val_loss',
                                          patience=args.patience)],
                 use_multiprocessing=True,
                 workers=args.workers)

    # Retrain the best model
    save_dst = args.save_path + '/best-fp.h5'
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    model.fit(train_dataset,
              epochs=10*args.epochs,
              validation_data=test_dataset,
              callbacks=[EarlyStopping('val_loss', patience=10*args.patience),
                         ModelCheckpoint(save_dst)],
              use_multiprocessing=True,
              workers=args.workers)

    # Evaluate the model
    model = load_model(save_dst)
    with h5py.File(args.dataset_path, 'r') as f:
        evaluate(model, test_dataset, f['y_test'][()],
                 save_path='%s/evaluation-fp.png' % args.save_path)

    # Load AutoQ configuration
    with open(args.autoq_config, 'r') as f:
        run_config = json.load(f)
    run_config["output_dir"] = args.save_path + "/autoq"

    # Set blocks
    if 'blocks' not in run_config:
        blocks = []
        l_names = model.layers[1:-1]
        l_names = [x.name.split('_')[-1] for x in l_names]
        l_names = np.unique(np.array(l_names))
        l_names = l_names[l_names != 'flatten']
        for ln in l_names:
            blocks.append('^.*_%s*.$' % ln)
        blocks.append('^output_softmax$')
        run_config["blocks"] = blocks

    # Recompile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=metrics)

    # Run AutoQ
    autoq = AutoQKerasScheduler(model, metrics, {}, debug=0, **run_config)

    autoq.fit(train_dataset,
              epochs=args.epochs,
              steps_per_epoch=args.steps,
              validation_data=test_dataset,
              validation_steps=args.steps,
              use_multiprocessing=True,
              workers=args.workers)

    # Retrain the best model
    save_dst = args.save_path + '/best-aq.h5'
    model = autoq.get_best_model()

    model.fit(train_dataset,
              epochs=10*args.epochs,
              validation_data=test_dataset,
              callbacks=[EarlyStopping('val_loss', patience=10*args.patience),
                         ModelCheckpoint(save_dst)],
              use_multiprocessing=True,
              workers=args.workers)

    # Evaluate the qmodel
    model = load_qmodel(save_dst)
    with h5py.File(args.dataset_path, 'r') as f:
        evaluate(model, test_dataset, f['y_test'][()],
                 save_path='%s/evaluation-aq.png' % args.save_path)
