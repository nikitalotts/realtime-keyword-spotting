import json
import os
import torch.nn.functional as F
import requests
import time
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.models.MatchboxNet import MatchboxNet
from src.logger import logger
from src.KeywordSpottingDataset import KeywordSpottingDataset

# Define constants
SAMPLE_RATE = 16000  # The sample rate of the audio data
HOP_LENGTH = 0.5  # The amount of second on which window will move each iteration.
WINDOW_SIZE = 1  # The amount of seconds in every frame
OUTPUT_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), './../outputs')  # The path for the outpu directory
OUTPUT_RADIO_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, 'radio.txt')  # The output path for the radio file
TEMP_AUDIO_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, 'temp.mp3')  # The temporary file path for the audio file


class KeywordSpotter:
    """
    A class for detecting wake words in audio files.

    Parameters:
        model_path (str): The path to a pre-trained model.

    Attributes:
        SAMPLE_RATE (int): The sample rate of the audio files.
        hop_length (float): The hop length in seconds.
        window_size (float): The window length in seconds.
        model (MatchboxNet): The pre-trained model used for prediction.
        training_data (None or dict): The training data used for the pre-trained model.

    Methods:
        check_folders(): Creates the output folder if it doesn't exist.
        detect_wake_words(audio_file_path): Detects wake words in an audio file and returns a list of times where the wake word was detected.

    Example:
        spotter = KeywordSpotter('models/matchboxnet.pth')
        res = spotter.detect_wake_words('audio_files/sample.wav')
        print(res)
    """

    def __init__(self, model_path):
        """
        Initializes the KeywordSpotter class with the necessary parameters and loads the pre-trained model.

        Parameters:
            model_path (str): The path to a pre-trained model.
        """
        self.SAMPLE_RATE = SAMPLE_RATE  # The sample rate of the audio files.
        self.hop_length = HOP_LENGTH  # The hop length in seconds.
        self.window_size = WINDOW_SIZE  # The window length in seconds.
        self.model = MatchboxNet(B=3, R=2, C=64, bins=64, NUM_CLASSES=2)
        self.model.load_state_dict(torch.load(model_path))
        self.model.float().to("cpu")
        self.training_data = None

    def check_folders(self):
        """
        Creates the output folder if it doesn't exist.
        """
        if not os.path.exists(OUTPUT_FOLDER_PATH):
            os.mkdir(OUTPUT_FOLDER_PATH)

    def detect_wake_words(self, audio_file_path):
        """
       Detects wake words in an audio file and returns a list of times where the wake word was detected.

       Parameters:
           audio_file_path (str): The path to the audio file.

       Returns:
           list: A list of times (in seconds) where the wake word was detected.

       """
        res = []
        hop_length = int(self.hop_length * self.SAMPLE_RATE)  # 0.5 second
        output_file = os.path.join(OUTPUT_FOLDER_PATH, os.path.basename(audio_file_path).replace(".wav", ".txt"))
        with open(output_file, 'w') as f:
            pass  # clear the file

        # load file
        y, sr = librosa.load(audio_file_path, sr=self.SAMPLE_RATE)

        # add paddings if need
        n_pad = self.window_size - (len(y) % self.window_size)
        y_padded = np.concatenate((y, np.zeros(n_pad)))
        sec = self.hop_length
        frames = int(y.shape[0] / hop_length)
        self.model.eval()
        with torch.no_grad():
            for i in range(frames):
                window = y_padded[int(sec * self.SAMPLE_RATE): int(
                    sec * self.SAMPLE_RATE + (self.SAMPLE_RATE * self.window_size))]
                tensor = self.convert_audio_to_tensor(window)
                prediction = self.model(tensor)
                probs = F.softmax(prediction, dim=1)
                preds = torch.argmax(probs, dim=1).item()
                print(f"sec {sec}:", preds)
                threshold = 0.5
                if preds > threshold:
                    res.append(sec)
                    with open(output_file, 'a') as f:
                        f.write(
                            os.path.join(OUTPUT_FOLDER_PATH, f'{os.path.basename(audio_file_path)}:{sec + 1:.0f}\n'))
                    cut_name = (os.path.join(OUTPUT_FOLDER_PATH,
                                             os.path.basename(audio_file_path).replace(".wav", f"_{sec + 1:.0f}.wav")))
                    sf.write(cut_name, y_padded[int(sec * SAMPLE_RATE) + (1 * SAMPLE_RATE): int(
                        sec * SAMPLE_RATE + (SAMPLE_RATE * 2.5))], samplerate=SAMPLE_RATE)
                sec += self.hop_length
        return res

    def convert_audio_to_tensor(self, audio):
        """Converts an audio signal to an MFCC tensor.

        Args:
            audio (ndarray): Input audio signal.

        Returns:
            Tensor: MFCC tensor of shape (1, 64, T) where T is the number of frames.
        """
        audio = self.model.emphasis(audio)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.SAMPLE_RATE, n_mfcc=64)
        inputs = self.model.padding(torch.from_numpy(mfcc.reshape(1, 64, -1)), 128)
        tensor = inputs.reshape(1, 64, -1).float()
        return tensor

    def process_radio_stream(self, radio_url):
        """Processes a radio stream for keyword detection.

        Args:
            radio_url (str): URL of the radio stream.

        Returns:
            None
        """
        try:
            output_file = OUTPUT_RADIO_FILE_PATH
            with open(output_file, 'w') as f:
                f.write(f'Listening to {radio_url}:\n')

            start_flag = True
            while not start_flag:
                try:
                    audio = requests.get(radio_url, stream=True)
                    audio.raise_for_status()
                    start_flag = True
                except:
                    time.sleep(10)

            audio = requests.get(radio_url, stream=True)
            audio.raise_for_status()

            streamed_audio = np.array([])
            timing = 0
            next_allowed_timing = 0
            keyword_timing = 0
            keyword = False

            self.model.eval()
            for chunk in audio.iter_content(chunk_size=8192):
                if chunk:
                    audio_file = open(TEMP_AUDIO_FILE_PATH, 'wb+')
                    audio_file.write(chunk)
                    audio_file.close()

                    try:
                        my_signal, sample_rate = librosa.load(TEMP_AUDIO_FILE_PATH, sr=self.SAMPLE_RATE)
                    except Exception as e:
                        print(e)
                        continue

                    timing += my_signal.shape[0] / sample_rate
                    streamed_audio = np.concatenate([streamed_audio, my_signal])

                    # get 1st second then start to predict
                    while next_allowed_timing + 1 < timing:
                        print(f"listned {next_allowed_timing + 1} seconds", end=": ")
                        frame = streamed_audio[int((next_allowed_timing) * sample_rate):int((next_allowed_timing + 1) * sample_rate)]
                        frame = self.convert_audio_to_tensor(frame)
                        with torch.no_grad():
                            prediction = self.model(frame)
                            probs = F.softmax(prediction, dim=1)
                            pred = torch.argmax(probs, dim=1).item()
                        threshold = 0.5
                        if pred > threshold:
                            print(f"found on {keyword_timing} second; pred: {pred}")
                            keyword = True
                            keyword_timing = next_allowed_timing
                        else:
                            print(f"nothing found; pred: {pred}")

                        # move window for hop_length to predict next frame
                        next_allowed_timing += self.hop_length
                    if keyword and (keyword_timing + 3 < timing):
                        print(f"DETECTED!!!")
                        logger.info(f"DETECTED ON {keyword_timing} second")
                        with open(output_file, 'a') as f:
                            f.write(f'{radio_url}:{int(next_allowed_timing + 1):.0f}\n')

                        cut_name = os.path.join(OUTPUT_FOLDER_PATH, f'radio_{int(keyword_timing)}.wav')
                        keyword_frame = streamed_audio[
                                       int((keyword_timing - 1) * sample_rate):int((keyword_timing + 3) * sample_rate)]
                        sf.write(cut_name, keyword_frame, samplerate=self.SAMPLE_RATE)
                        keyword = False
        except KeyboardInterrupt:
            if os.path.exists(TEMP_AUDIO_FILE_PATH):
                os.remove(TEMP_AUDIO_FILE_PATH)
            print("removed cache")

    def choose_device(self):
        """
        Method to choose the device for training the model.

        Returns:
            device (str): "cuda" if a GPU is available, otherwise "cpu".
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_data(self, data_path):
        """
        Method to load the training data from a JSON file and store it in the KeywordSpotter object.

        Args:
            data_path (str): path to the JSON file containing the training data.

        Returns:
            None
        """
        f = open(data_path)
        data = json.load(f)
        for old_key in list(data.keys()):
            new_key = data_path.replace(
                "data.json", old_key)
            data[new_key] = data.pop(old_key)
        self.training_data = data

    def reload_model(self, model_path=None):
        """
        Method to reload the MatchboxNet model for training.

        Args:
            model_path (str): path to the saved model file (default: None).

        Returns:
            None
        """
        self.model = MatchboxNet(B=3, R=2, C=64, bins=64, NUM_CLASSES=2)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        self.model.float().to("cpu")

    def prepare(self, data_path, batch_size, test_size=0.2):
        """
       Method to prepare the training and validation data for training the model.

       Args:
           data_path (str): path to the JSON file containing the training data.
           batch_size (int): batch size for training the model.
           test_size (float): fraction of the data to use for validation (default: 0.2).

       Returns:
           train_dataloader (DataLoader): dataloader for the training data.
           val_dataloader (DataLoader): dataloader for the validation data.
           criterion (CrossEntropyLoss): loss function for training the model.
           optimizer (Adam): optimizer for training the model.
           device (str): "cuda" if a GPU is available, otherwise "cpu".
       """
        self.load_data(data_path)
        device = self.choose_device()

        # prepare data
        file_paths = [path for path in self.training_data.keys()]
        labels = [label for label in self.training_data.values()]
        X_train_paths, X_val_paths, y_train, y_val = train_test_split(file_paths, labels, test_size=test_size,
                                                                      random_state=42, shuffle=True, stratify=labels)
        # create train and validation datasets
        train_dataset = KeywordSpottingDataset(X_train_paths, y_train)
        val_dataset = KeywordSpottingDataset(X_val_paths, y_val)

        # create train and validation dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        return train_dataloader, val_dataloader, criterion, optimizer, device

    def run_epochs(self, model, loader, criterion, optimizer, device):
        """
        Trains the given model for one epoch on the given data loader using the given criterion and optimizer.

        Args:
            model (nn.Module): The model to train.
            loader (DataLoader): The data loader to use for training.
            criterion (nn.Module): The loss criterion to use for training.
            optimizer (optim.Optimizer): The optimizer to use for training.
            device (str): The device to use for training.

        Returns:
            tuple: A tuple containing the average loss and accuracy achieved during the epoch.
        """
        model.to(device)
        model.train()
        running_loss = 0.0
        acc = 0
        count = 0
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.float().to(device)
            targets = targets.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_acc = Accuracy(num_classes=2, compute_on_step=False, dist_sync_on_step=False, task='binary')(
                predicted.cpu(), targets.cpu())
            acc += running_acc
            count = i
        accuracy = acc / (count + 1)
        return running_loss / len(loader), accuracy

    def validate(self, model, loader, criterion, device):
        """
        Runs validation on the given model using the given data loader and criterion.

        Args:
            model (nn.Module): The model to validate.
            loader (DataLoader): The data loader to use for validation.
            criterion (nn.Module): The loss criterion to use for validation.
            device (str): The device to use for validation.

        Returns:
            tuple: A tuple containing the average loss and accuracy achieved during validation.
        """
        model.eval()
        model.to(device)
        running_loss = 0.0
        acc = 0
        count = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(loader):
                inputs = inputs.float().to(device)
                targets = targets.type(torch.LongTensor).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                running_acc = Accuracy(num_classes=2, compute_on_step=False, dist_sync_on_step=False, task='binary')(
                    predicted.cpu(), targets.cpu())
                print('training running accuracy', running_acc)
                acc += running_acc
                count = i
            accuracy = acc / (count + 1)
        return running_loss / len(loader), accuracy

    def train(self, data_path, batch_size, n_epochs, test_size):
        """
        Trains the MatchboxNet model on the provided data.

        Args:
            data_path (str): Path to the JSON data file.
            batch_size (int): The batch size to be used for training.
            n_epochs (int): Number of epochs to train the model.
            test_size (float): The fraction of data to be used for validation.

        Returns:
            None

        """
        train_dataloader, val_dataloader, criterion, optimizer, device = self.prepare(data_path, batch_size, test_size)
        self.reload_model()
        for epoch in range(n_epochs):
            train_loss, train_acc = self.run_epochs(self.model, train_dataloader, criterion, optimizer, device)
            val_loss, val_acc = self.validate(self.model, val_dataloader, criterion, device)
            print(
                f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

        model_path = f'./src/models/store/MatchboxNet_{n_epochs}_epochs.pth'
        torch.save(self.model.state_dict(), model_path)
        self.reload_model(model_path)
        logger.info("model trained")

    def evaluate(self, data_path, batch_size):
        """
        Evaluates the MatchboxNet model on the provided data.

        Args:
           data_path (str): Path to the JSON data file.
           batch_size (int): The batch size to be used for evaluation.

        Returns:
           None

        """
        _, val_dataloader, criterion, optimizer, device = self.prepare(data_path, batch_size)
        val_loss, val_acc = self.validate(self.model, val_dataloader, criterion, device)
        print(f"Evaluating results: Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        logger.info("model evaluated")
