import configparser
import h5py
import numpy as np
import os
import requests
import shutil
import tarfile


class PrepareDataset():

    def __init__(self, config_path):

        # Load configuration
        self.config = configparser.RawConfigParser()
        self.config.read(config_path)

        self.X_data_shape = (100, 100, 1)
        self.y_data_shape = (5,)

    def cleanup(self, tmpdir):
        shutil.rmtree(tmpdir)

    def create_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def download_data(self):
        train_dataset = self.config.get('source', 'TrainDataset')
        test_dataset = self.config.get('source', 'TestDataset')
        tmp_dir = self.config.get('store', 'TmpDirectory')
        save_dir = self.config.get('store', 'SaveDirectory')

        train_path = '%s/train.tar.gz' % tmp_dir
        test_path = '%s/test.tar.gz' % tmp_dir

        # Creating directories
        self.create_directory(tmp_dir)
        self.create_directory(save_dir)

        # Fetch files from url
        self.fetch(train_dataset, train_path)
        self.fetch(test_dataset, test_path)

        # Unarchive fetched files
        self.unarchive(train_path, tmp_directory)
        self.unarchive(test_path, tmp_directory)

        self.process_data('%s/train' % tmp_directory,
                          save_directory, 'train')

        self.cleanup(tmp_dir)

    def fetch(self, url, store):
        r = requests.get(url)
        with open(store, 'wb') as f:
            f.write(r.content)

    def process_data(self, in_dir, out_dir, set_type):
        X_set = 'X_%s' % set_type
        y_set = 'y_%s' % set_type

        with h5py.File('%s/jetimg.h5' % out_dir, 'w') as h5:
            row = 0
            for fname in os.listdir(in_dir):
                infile = os.path.join(in_dir, fname)
                with h5py.File(infile, "r") as f:
                    X = np.array(f.get("jetImage"))
                    X = X[..., np.newaxis]
                    y = np.array(f.get('jets')[0:, -6:-1])

                if row == 0:
                    h5.create_dataset(X_set, dtype=np.dtype('float64'),
                                      shape=(0,) + self.X_data_shape,
                                      maxshape=(None,) + self.X_data_shape)
                    h5.create_dataset(y_set, dtype=np.dtype('float64'),
                                      shape=(0,) + self.y_data_shape,
                                      maxshape=(None,) + self.y_data_shape)

                crows = y.shape[0]

                h5[X_set].resize(((row+crows,) + self.X_data_shape))
                h5[X_set][row:row+crows, :] = X
                h5[y_set].resize(((row+crows,) + self.y_data_shape))
                h5[y_set][row:row+crows, :] = y

                row += crows

    def unarchive(self, fname, tmpdir):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path=tmpdir)
        tar.close()


pd = PrepareDataset(r'./config.txt')
pd.download_data()
