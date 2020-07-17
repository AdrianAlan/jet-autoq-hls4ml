import configparser
import requests
import tarfile


class PrepareDataset():

    def __init__(self, config_path):

        # Load configuration
        self.config = configparser.RawConfigParser()
        self.config.read(config_path)

    def download_data(self):
        train_dataset = self.config.get('source', 'TrainDataset')
        test_dataset = self.config.get('source', 'TestDataset')
        tmp_directory = self.config.get('store', 'TmpDirectory')

        train_path = '%s/train.tar.gz' % tmp_directory
        test_path = '%s/test.tar.gz' % tmp_directory

        # Fetch files from url
        self.fetch(train_dataset, train_path)
        self.fetch(test_dataset, test_path)

        # Unarchive fetched files
        self.unarchive(train_path, tmp_directory)
        self.unarchive(test_path, tmp_directory)

    def fetch(self, url, store):
        r = requests.get(url)
        with open(store, 'wb') as f:
            f.write(r.content)

    def unarchive(self, fname, tmpdir):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path=tmpdir)
        tar.close()


pd = PrepareDataset(r'./config.txt')
pd.download_data()
