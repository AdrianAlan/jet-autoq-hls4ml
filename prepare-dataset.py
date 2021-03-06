import h5py
import numpy as np
import logging
import os
import requests
import shutil
import tarfile

from configparser import ConfigParser, ExtendedInterpolation


class PrepareDataset():

    def __init__(self, config_path):

        # Load configuration
        self.parse_config(config_path)

    def cleanup(self, tmpdir):
        shutil.rmtree(tmpdir)

    def create_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def execute(self):

        logging.basicConfig(level=logging.INFO)

        logging.info('Creating working directories')
        self.create_directory(self.tmp_dir)
        self.create_directory(self.save_dir)

        logging.info('Fetching the dataset 1/2')
        self.fetch_data(self.train_dataset, self.tmp_train_path)
        logging.info('Fetching the dataset 2/2')
        self.fetch_data(self.test_dataset, self.tmp_test_path)

        logging.info('Unarchiving working files')
        self.unarchive(self.tmp_train_path, self.tmp_dir)
        self.unarchive(self.tmp_test_path, self.tmp_dir)

        logging.info('Processing data')
        self.process_data(self.train_dataset_dir, self.save_dir, 'train',
                          self.X_data_shape, self.y_data_shape)
        self.process_data(self.test_dataset_dir, self.save_dir, 'test',
                          self.X_data_shape, self.y_data_shape)

        logging.info('Final cleaning')
        self.cleanup(self.tmp_dir)

        logging.info('Done')

    def fetch_data(self, url, store):
        r = requests.get(url)
        with open(store, 'wb') as f:
            f.write(r.content)

    def parse_config(self, config_path):
        self.config = ConfigParser(interpolation=ExtendedInterpolation(),
                                   converters={'tuple': self.parse_int_tuple})
        self.config.read(config_path)
        self.train_dataset = self.config.get('source', 'TrainDataset')
        self.train_dataset_dir = self.config.get('source', 'TrainDirectory')
        self.test_dataset = self.config.get('source', 'TestDataset')
        self.test_dataset_dir = self.config.get('source', 'TestDirectory')

        self.tmp_dir = self.config.get('store', 'TmpDirectory')
        self.save_dir = self.config.get('store', 'SaveDirectory')
        self.tmp_train_path = self.config.get('store', 'TmpTrainPath')
        self.tmp_test_path = self.config.get('store', 'TmpTestPath')

        self.X_data_shape = self.config.gettuple('data', 'XShape')
        self.y_data_shape = self.config.gettuple('data', 'yShape')

    def parse_int_tuple(self, input):
        return tuple(int(k.strip()) for k in input[1:-1].split(','))

    def process_data(self, in_dir, out_dir, set_type,
                     X_data_shape, y_data_shape):
        X_set = 'X_%s' % set_type
        y_set = 'y_%s' % set_type

        with h5py.File('%s/jetimg.h5' % out_dir, 'a') as h5:
            row = 0
            for fname in os.listdir(in_dir):
                infile = os.path.join(in_dir, fname)
                with h5py.File(infile, "r") as f:
                    X = np.array(f.get("jetImage"))
                    X = X[..., np.newaxis]
                    y = np.array(f.get('jets')[0:, -6:-1])

                if row == 0:
                    h5.create_dataset(X_set, dtype=np.dtype('float64'),
                                      shape=(0,) + X_data_shape,
                                      maxshape=(None,) + X_data_shape)
                    h5.create_dataset(y_set, dtype=np.dtype('float64'),
                                      shape=(0,) + y_data_shape,
                                      maxshape=(None,) + y_data_shape)

                crows = y.shape[0]

                h5[X_set].resize(((row+crows,) + X_data_shape))
                h5[X_set][row:row+crows, :] = X
                h5[y_set].resize(((row+crows,) + y_data_shape))
                h5[y_set][row:row+crows, :] = y

                row += crows

    def unarchive(self, fname, tmpdir):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path=tmpdir)
        tar.close()


if __name__ == '__main__':
    pd = PrepareDataset(r'./config.txt')
    pd.execute()
