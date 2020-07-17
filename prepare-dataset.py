import configparser
import requests


class PrepareDataset():

    def __init__(self, config_path):

        # Load configuration
        self.config = configparser.RawConfigParser()
        self.config.read(config_path)

    def download_data(self):
        train_dataset = self.config.get('source', 'TrainDataset')
        test_dataset = self.config.get('source', 'TestDataset')
        tmp_directory = self.config.get('store', 'TmpDirectory')

        self.fetch(train_dataset, '%s/train.tar.gz' % tmp_directory)
        self.fetch(test_dataset, '%s/test.tar.gz' % tmp_directory)

    def fetch(self, url, store):
        r = requests.get(url)
        with open(store, 'wb') as f:
            f.write(r.content)


pd = PrepareDataset(r'./config.txt')
pd.download_data()
