import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

TRAINING_SAMPLE_RATE = 16000


class KeywordSpottingDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths) - 1

    def emphasis(self, audio, pre_emphasis=0.97):
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        return audio

    def crop_audio(self, audio, sr=16000):
        # Get length of audio in samples
        audio_len = audio.shape[0]

        silence_length = np.random.uniform(0.3, 0.5)
        # print("silence_length", silence_length)
        # Calculate max number of samples to replace with silence
        silence_len = int(silence_length * sr) # 10% of audio length

        in_start = np.random.choice([True, False])
        # print('instart', in_start)
        if in_start:
            start_idx = 0
            end_idx = start_idx + silence_len
        else:
            start_idx = audio_len - silence_len - 1
            end_idx = audio_len - 1

        # Replace audio segment with silence
        augmented_audio = audio.copy()
        augmented_audio[start_idx:end_idx] = 0.0

        return augmented_audio

    def add_noise(self, audio, noise_level=0.01):
        noise = np.random.normal(scale=noise_level, size=len(audio))
        return audio + noise

    def pitch_shift(self, audio, sr=16000):
        steps = [-3, -2, -1, 1, 2, 3]
        choice = np.random.choice(steps, 1)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=choice)

    def augment_audio(self, audio):
        # Randomly apply one or more augmentation methods
        methods = []
        if np.random.random() < 0.5:
            methods.append(self.crop_audio)
        if np.random.random() < 0.5:
            methods.append(self.add_noise)
        if np.random.random() < 0.5:
            methods.append(self.pitch_shift)

        # Apply selected augmentation methods to audio
        for method in methods:
            audio = method(audio)

        return audio

    def get_log_mel_spectrogram(self, audio):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=TRAINING_SAMPLE_RATE, n_fft=4096, hop_length=512, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = mel_spec_db.astype(np.float32)
        return mel_spec_db


    def normalize_spectrogram(self, mel_spec_db):
        mel_spec_db = librosa.util.normalize(mel_spec_db)
        return mel_spec_db

    def get_mfcc(self, audio):
        mfcc = librosa.feature.mfcc(y=audio, sr=TRAINING_SAMPLE_RATE, n_mfcc=40, n_fft=4096, hop_length=512)
        mfcc = mfcc.astype(np.float32)
        return mfcc

    def pad_mfccs(self, mfccs):
        # mfccs shape: (40, 63)
        pad_width = ((0, 128 - mfccs.shape[0]), (0, 0))
        padded_mfccs = np.pad(mfccs, pad_width, mode='constant')
        return padded_mfccs

    def merge_spec_and_mfcc(self, mel_spec_db, mfccs):
        spectrograms = torch.stack([torch.from_numpy(mel_spec_db), torch.from_numpy(mfccs)])
        return spectrograms

    def padding(self, batch, seq_len):
        if len(batch[0][0]) < seq_len:
            m = torch.nn.ConstantPad1d((0, seq_len - len(batch[0][0])), 0)
            batch = m(batch)
        return batch

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        audio, _ = librosa.load(audio_path, sr=TRAINING_SAMPLE_RATE)
        audio = self.emphasis(audio)
        audio = self.augment_audio(audio)
        mfcc = librosa.feature.mfcc(y=audio, sr=TRAINING_SAMPLE_RATE, n_mfcc=64)
        inputs = self.padding(torch.from_numpy(mfcc.reshape(1, 64, -1)), 128)
        tensor = inputs.reshape(64, -1)
        label = torch.tensor(label, dtype=torch.long)
        label.numpy()
        return tensor, label
