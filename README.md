# Sound Processing and Realtime Keyword Spotting

This project implements a system that can detect keywords in audio files or in radio streaming

The project was implemented in the 7th module of the 3rd year of [Higher IT School](https://hits.tsu.ru/), [Tomsk State University](https://www.tsu.ru/), [Tomsk](https://en.wikipedia.org/wiki/Tomsk).


## Models

The project use [MatchboxNet](https://arxiv.org/abs/2004.08531) model: an end-to-end neural network for speech command recognition

##  Installation

The project does not require installation, just download, unpack the files of this repository* and install requirements**.

\* *the repository does not contain the datasets on which the model was trained, if you want to use them, then download the archive from [here](https://drive.google.com/file/d/1ONZ8JSa93GXT8f6FAft7q7430lcU3kWl/view?usp=sharing) and unpack it to the root folder of the project*

\** Run following command in shell:

``` 
$ pip install -r requirements.txt
```

## User manual

### Datasets

In the learning process, a dataset was used , which consists of many .wav files(during the training process, three augmentations were applied to them: adding noise, pitching and crop) with a length of 1 second each 
and a description file in JSON format, which is a dictionary where the keys are the relative paths from this annotation file to the corresponding audio file, and the values are int values: 1 if this fragment contains a keyword and 0 if not.

***In this model, the keyword was "STONES"***

Further, in order for the model to train correctly and work according to the structure, the new datasets must match the original dataset:

#### Example

Dataset structure:
```
dataset
│   data.json 
│
└───clips
│   │   0.wav
│   │   1.wav
│   │   2.wav
|   |   ...
```

and `data.json` file should look like this***:

```
$ cat data.json

{"./clips/0.wav": 0, "./clips/1.wav": 0, "./clips/2.wav": 0}
```

***the paths specified in the `data.json` must be **relative paths** from data file


### CLI

The system works through the following CLI commands:
1. train:
```$ python path/to/model.py train --parameter=value```

```
Trains the KeywordSpotter model using the provided training data. Save trained model to models` store folder

Parameters:
    data_path (str): The file path of the JSON file containing the training data. Defaults to the constant value TRAINING_DATA(value: path to source train dataset).
    batch_size (int): The batch size to use during training. Defaults to the constant value BATCH_SIZE(value: 16).
    n_epochs (int): The number of epochs to train for. Defaults to the constant value N_EPOCHS(value: 5).
    test_size (float): The fraction of the data to use for testing. Defaults to the constant value TEST_SIZE(value: 0.5).
Returns:
    None
```

2. evaluate: ```$ python path/to/model.py evaluate --parameter=value```

```
Evaluates the performance of the KeywordSpotter model using the provided evaluation data and print current accuracy of model.

Parameters:
    data_path (str): The file path of the JSON file containing the evaluation data. Defaults to the constant value TRAINING_DATA(value: path to source train dataset).
    batch_size (int): The batch size to use during evaluation. Defaults to the constant value BATCH_SIZE(value: 16).
Returns:
    None
```

3. listen: ```$ python path/to/model.py listen --parameter=value```

```
Listens to a live radio stream and detects wake words using a pre-trained model, crop audio clips from stream which contain the keywords and store them in output folder End after 4 hours of listening(by default) or if user closed program by pressing the combination “CTRL + C”

Args:
    radio (str, optional): The URL of the radio stream to listen to. Defaults to `'https://radio.maslovka-home.ru/soundcheck'`.
    model_path (str, optional): The path to the pre-trained model. Defaults to `None`.
Returns:
    None
```

4. find: ```$ python path/to/model.py find --parameter=value```

```
Finds wake words in an audio file using a pre-trained model, crop word following the keywords and store them in output folder

Args:
   audio (str, optional): The path to the audio file to process. Defaults to `'./training_data/data/thanos_message.wav'`.
    model_path (str, optional): The path to the pre-trained model. Defaults to `None`.
Returns:
   None
```

### References
* [TorchKWS](https://github.com/swagshaw/TorchKWS)
* [SPEECH COMMAND CLASSIFICATION WITH TORCHAUDIO](https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html)
* [A PyTorch implementation of MatchboxNet from Scratch](https://github.com/dominickrei/MatchboxNet)
* [Build a deep neural network for the keyword spotting (KWS) task with nnAudio GPU audio processing](https://towardsdatascience.com/build-a-deep-neural-network-for-the-keyword-spotting-kws-task-with-nnaudio-gpu-audio-processing-95b50018aaa8)

### Author
*Nikita Lotts, 3rd grade student in Tomsk State University (Tomsk)*