import os
import torch.nn.functional as F
import torch
import requests
import time
import numpy as np
import librosa
import soundfile as sf

from src.models.store.matchboxnet import MatchboxNet
from src.logger import logger

from scipy.linalg import _fblas

SAMPLE_RATE = 16000
HOP_LENGTH = 0.5
WINDOW_SIZE = 1
OUTPUT_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), './../outputs')
OUTPUT_RADIO_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, 'radio.txt')
TEMP_AUDIO_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, 'temp.mp3')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.111 Safari/537.36'}


class KeywordSpotter:
    def __init__(self, model_path):
        self.SAMPLE_RATE = SAMPLE_RATE
        self.hop_length = HOP_LENGTH # hop length in seconds
        self.window_size = WINDOW_SIZE # window length in seconds
        self.model = MatchboxNet(B=3, R=2, C=64, bins=64, NUM_CLASSES=2)
        self.model.load_state_dict(torch.load(model_path))
        self.model.float().to("cpu")

    def check_folders(self):
        if not os.path.exists(OUTPUT_FOLDER_PATH):
            os.mkdir(OUTPUT_FOLDER_PATH)

    def detect_wake_words(self, audio_file_path):

        res = []

        hop_length = int(self.hop_length * self.SAMPLE_RATE)  # 0.5 second

        output_file = os.path.join(OUTPUT_FOLDER_PATH, os.path.basename(audio_file_path).replace(".wav", ".txt"))
        with open(output_file, 'w') as f:
            pass  # clear the file

        # Load the input .wav file
        y, sr = librosa.load(audio_file_path, sr=self.SAMPLE_RATE)

        # Pad the input signal to ensure that we get predictions for the entire file
        n_pad = self.window_size - (len(y) % self.window_size)
        y_padded = np.concatenate((y, np.zeros(n_pad)))

        sec = self.hop_length

        frames = int(y.shape[0] / hop_length)
        # Loop through the sliding windows of the input signal

        self.model.eval()
        with torch.no_grad():
            for i in range(frames):

                # for i in range(0, len(y_padded) - window_size + 1, hop_length):

                # Get the current window
                window = y_padded[int(sec * self.SAMPLE_RATE): int(sec * self.SAMPLE_RATE + (self.SAMPLE_RATE * self.window_size))]

                window = self.model.emphasis(window)

                mfcc = librosa.feature.mfcc(y=window, sr=self.SAMPLE_RATE, n_mfcc=64)
                inputs = self.model.padding(torch.from_numpy(mfcc.reshape(1, 64, -1)), 128)
                tensor = inputs.reshape(1, 64, -1).float()

                # Make a prediction with the PyTorch model
                prediction = self.model(tensor)

                probs = F.softmax(prediction, dim=1)
                # Get predicted class indices
                preds = torch.argmax(probs, dim=1).item()

                # Get predicted class indices
                # print(f"i: {i}; sec {sec}:", preds)
                print(f"sec {sec}:", preds)

                # Check if the prediction is above a certain threshold
                threshold = 0.5
                if preds > threshold:
                    res.append(sec)
                    # If the prediction is above the threshold, log the timing
                    # time_start = i / sr - 1  # start 1 sec earlier
                    # time_end = time_start + 4  # end 4 sec after the detection
                    with open(output_file, 'a') as f:
                        f.write(os.path.join(OUTPUT_FOLDER_PATH, f'{os.path.basename(audio_file_path)}:{sec + 1:.0f}\n'))

                    # Cut the .wav file with that detection and save it
                    cut_name = (os.path.join(OUTPUT_FOLDER_PATH, os.path.basename(audio_file_path).replace(".wav", f"_{sec + 1:.0f}.wav")))

                    # такую запись при радио
                    # sf.write(cut_name,
                    #          y_padded[int(sec * self.SAMPLE_RATE) - self.SAMPLE_RATE: int(sec * self.SAMPLE_RATE + (self.SAMPLE_RATE * 3))],
                    #          samplerate=self.SAMPLE_RATE)

                    sf.write(cut_name, y_padded[int(sec * SAMPLE_RATE) + (1 * SAMPLE_RATE): int(sec * SAMPLE_RATE + (SAMPLE_RATE * 2.5)) ], samplerate=SAMPLE_RATE)
                sec += self.hop_length
        return res

    def process_radio_stream(self, radio_url):
        try:
            output_file = OUTPUT_RADIO_FILE_PATH
            with open(output_file, 'w') as f:
                f.write(f'Listening to {radio_url}:\n')

            start_flag = True
            while not start_flag:
                try:
                    audio = requests.get(radio_url, stream=True, headers=HEADERS)
                    audio.raise_for_status()
                    start_flag = True
                except:
                    time.sleep(10)

            audio = requests.get(radio_url, stream=True, headers=HEADERS)
            audio.raise_for_status()

            signal = np.array([])
            time_line = 0
            next_predict_ind = 0
            found_flag = False
            found_ind = 0

            self.model.eval()
            for chunk in audio.iter_content(chunk_size=8192):
                if chunk:
                    audio_file = open(TEMP_AUDIO_FILE_PATH, 'wb+')
                    audio_file.write(chunk)
                    audio_file.close()

                    try:
                        my_signal, sample_rate = librosa.load(TEMP_AUDIO_FILE_PATH, sr=self.SAMPLE_RATE)
                        # print("my_signal,sample_rate", sample_rate)
                    except Exception as e:
                        print(e)
                        continue

                    time_line += my_signal.shape[0] / sample_rate
                    signal = np.concatenate([signal, my_signal])

                    # получаем первую секунду - потом начинаем предсказывать
                    while next_predict_ind + 1 < time_line:
                        print(f"listned {next_predict_ind + 1} seconds", end=": ")
                        sample = signal[int((next_predict_ind) * sample_rate):int((next_predict_ind + 1) * sample_rate)]

                        sample = self.model.emphasis(sample)

                        mfcc = librosa.feature.mfcc(y=sample, sr=self.SAMPLE_RATE, n_mfcc=64)
                        inputs = self.model.padding(torch.from_numpy(mfcc.reshape(1, 64, -1)), 128)
                        tensor = inputs.reshape(1, 64, -1).float()

                        with torch.no_grad():
                            prediction = self.model(tensor)
                            probs = F.softmax(prediction, dim=1)
                            pred = torch.argmax(probs, dim=1).item()

                        threshold = 0.5
                        if pred > threshold:
                            print(f"found on {found_ind} second; pred: {pred}")
                            found_flag = True
                            found_ind = next_predict_ind
                        else:
                            print(f"nothing found; pred: {pred}")

                        # сдвигаем окно на 0.5
                        next_predict_ind += self.hop_length
                    if found_flag and (found_ind + 3 < time_line):
                        print(f"DETECTED!!!")
                        logger.info(f"DETECTED ON {found_ind} second")
                        with open(output_file, 'a') as f:
                            f.write(f'{radio_url}:{next_predict_ind + 1:.0f}\n')

                        cut_name = os.path.join(OUTPUT_FOLDER_PATH , f'radio_{found_ind}.wav')
                        found_signal = signal[int((found_ind - 1) * sample_rate):int((found_ind + 5) * sample_rate)]
                        sf.write(cut_name, found_signal, samplerate=self.SAMPLE_RATE)
                        found_flag = False

        except KeyboardInterrupt:
            print("removed cache")



