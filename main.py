import os

import fire
from src.KeywordSpotter import KeywordSpotter
from src.logger import logger

TRAINING_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), './dataset/data.json')
BATCH_SIZE = 16
N_EPOCHS = 1
TEST_SIZE = 0.3


class CliWrapper(object):
    """
    This is a wrapper class for ODasModel model instance
    to use model with CLI commands
    """

    def __init__(self, ):
        self.spotter = None
        self.default_model = './src/models/store/model_9_60_1.pth'

    def __load_model(self, model_path=None):
        if model_path is None:
            model_path = self.default_model
        self.spotter = KeywordSpotter(model_path)
        self.spotter.check_folders()
        logger.info(f"CliWrapper object inited")

    def find(self, audio='./training_data/data/thanos_message.wav', model_path=None):
        self.__load_model(model_path)
        results = self.spotter.detect_wake_words(os.path.join(os.path.dirname(os.path.abspath(__file__)), audio))
        print(f'Keywords detected on {", ".join([str(x) for x in results])} seconds')
        # logger.info(f'Keywords detected on {", ".join([str(x) for x in results])} seconds')

    def listen(self, radio='https://radio.maslovka-home.ru/soundcheck', model_path=None):
        self.__load_model(model_path)
        self.spotter.process_radio_stream(radio)

    def listen_real(self, radio='http://radio.maslovka-home.ru/thanosshow', model_path=None):
        self.__load_model(model_path)
        self.spotter.process_radio_stream(radio)

    def train(self, data_path=TRAINING_DATA, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, test_size=TEST_SIZE):
        self.__load_model()
        self.spotter.train(data_path, batch_size, n_epochs, test_size)

    def evaluate(self, data_path=TRAINING_DATA, batch_size=BATCH_SIZE):
        self.__load_model()
        self.spotter.evaluate(data_path, batch_size)


if __name__ == "__main__":
    """Method that run Google's python-fire on CliWrapper class
    to start support of CLI commands"""

    fire.Fire(CliWrapper)
