import configparser


class PrepareDataset():

    def __init__(self, config_path):

        # Load configuration
        self.config = configparser.RawConfigParser()
        self.config.read(config_path)


pd = PrepareDataset(r'./config.txt')
