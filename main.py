import os
import fire
from src.KeywordSpotter import KeywordSpotter
from src.logger import logger

TRAINING_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), './dataset/data.json')  # Path to the
# training data file containing audio and labels in JSON format
BATCH_SIZE = 16  # Number of samples in each batch of data during training
N_EPOCHS = 1  # Number of times the entire training dataset is passed through the model during training
TEST_SIZE = 0.3  # The proportion of the training data to be used as the validation set during training


class CliWrapper(object):
    """
    This class is a CLI (Command Line Interface) wrapper for the `KeywordSpotter` class.
    It allows the user to interact with the `KeywordSpotter` class using command line arguments.
    """

    def __init__(self, ):
        """
        Initializes a new `CliWrapper` instance.

        Args:
            None

        Returns:
            None
        """
        self.spotter = None
        self.default_model = './src/models/store/MatchboxNet_1_sec.pth'

    def __load_model(self, model_path=None):
        """
        Loads a pre-trained model from a given path or the default path.

        Args:
            model_path (str, optional): The path to the pre-trained model. Defaults to `None`.

        Returns:
            None
        """
        if model_path is None:
            model_path = self.default_model
        self.spotter = KeywordSpotter(model_path)
        self.spotter.check_folders()
        logger.info(f"CliWrapper object inited")

    def find(self, audio='./training_data/data/thanos_message.wav', model_path=None):
        """
        Finds wake words in an audio file using a pre-trained model.

        Args:
           audio (str, optional): The path to the audio file to process. Defaults to `'./training_data/data/thanos_message.wav'`.
           model_path (str, optional): The path to the pre-trained model. Defaults to `None`.

        Returns:
           None
        """
        self.__load_model(model_path)
        results = self.spotter.detect_wake_words(os.path.join(os.path.dirname(os.path.abspath(__file__)), audio))
        print(f'Keywords detected on {", ".join([str(x) for x in results])} seconds')
        # logger.info(f'Keywords detected on {", ".join([str(x) for x in results])} seconds')

    def listen(self, radio='https://radio.maslovka-home.ru/soundcheck', model_path=None):
        """
        Listens to a live radio stream and detects wake words using a pre-trained model.

        Args:
            radio (str, optional): The URL of the radio stream to listen to. Defaults to `'https://radio.maslovka-home.ru/soundcheck'`.
            model_path (str, optional): The path to the pre-trained model. Defaults to `None`.

        Returns:
            None
        """
        self.__load_model(model_path)
        self.spotter.process_radio_stream(radio)

    def listen_real(self, radio='http://radio.maslovka-home.ru/thanosshow', model_path=None):
        """
        Listens to a live radio stream and detects wake words using a pre-trained model. The same method that `listen`
        but only for one specified radio station

        Args:
           radio (str, optional): The URL of the radio stream to listen to. Defaults to `'http://radio.maslovka-home.ru/thanosshow'`.
           model_path (str, optional): The path to the pre-trained model. Defaults to `None`.

        Returns:
           None
        """
        self.__load_model(model_path)
        self.spotter.process_radio_stream(radio)

    def train(self, data_path=TRAINING_DATA, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, test_size=TEST_SIZE):
        """
        Trains the KeywordSpotter model using the provided training data.

        Parameters:
            data_path (str): The file path of the JSON file containing the training data.
                Defaults to the constant value TRAINING_DATA.
            batch_size (int): The batch size to use during training. Defaults to the constant value BATCH_SIZE.
            n_epochs (int): The number of epochs to train for. Defaults to the constant value N_EPOCHS.
            test_size (float): The fraction of the data to use for testing. Defaults to the constant value TEST_SIZE.

        Returns:
            None
        """
        self.__load_model()
        self.spotter.train(data_path, batch_size, n_epochs, test_size)

    def evaluate(self, data_path=TRAINING_DATA, batch_size=BATCH_SIZE):
        """
        Evaluates the performance of the KeywordSpotter model using the provided evaluation data.

        Parameters:
            data_path (str): The file path of the JSON file containing the evaluation data.
                Defaults to the constant value TRAINING_DATA.
            batch_size (int): The batch size to use during evaluation. Defaults to the constant value BATCH_SIZE.

        Returns:
            None
        """
        self.__load_model()
        self.spotter.evaluate(data_path, batch_size)


if __name__ == "__main__":
    """
    This block of code checks if the script is being run as the main program, and if it is,
    it creates an instance of the CliWrapper class and uses the Google's python-fire library
    to enable CLI commands.
    """
    fire.Fire(CliWrapper)
