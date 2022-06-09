"""
A script to create and teach the model that we will use.

The model will be stored in models/RTF.pickle (RTF for Random Tree Forest)

Author : Alexandre SCHŒPP https://github.com/Alex-S-H-P/
"""


import os
import pickle
import typing
from typing import TextIO

import jams
import librosa
import numpy as np

from codebase.utillib.embedder import Embedder

try:
    import codebase.path_handler as path_handler
    import codebase.classify as classify
except ModuleNotFoundError:
    import path_handler
    import classify
finally:
    from sklearn.ensemble import RandomForestClassifier

annotations_path = path_handler.path_to_specific_dataset("annotations")
musics_path = path_handler.path_to_specific_dataset("musics")


def build_model() -> RandomForestClassifier:
    """
    Constructs a new instance of the model
    :return: model the new model
    """
    model = RandomForestClassifier(n_jobs=-1)
    return model


def getAnnotation(timeStamp: float, jamFile: jams.JAMS) -> str:
    """
    Gets the annotation at a given time.

    :param timeStamp: float, the time, in seconds, at which we are looking
    :param jamFile: JAMS, the file that stores the annotation
    :return: value, the value of the note at the given timeStamp
    """
    ic = jamFile.search(namespace="chord")
    for annotation in ic[1]["data"]:
        if annotation.time < timeStamp < annotation.time + annotation.duration:
            return annotation.value


def custom_generator(x_or_y: str) -> typing.Generator[list[float], None, None]:
    """
    A custom generator that can de-pickle storage file and handle training the model without leaking memory
    :param x_or_y: "x" if you want the input FFT decomposition. "y" if you want the output annotation.
    :return: a generator that the model can be trained on
    """
    assert x_or_y.lower() in ["x", "y"]
    returnX = x_or_y.lower() == "x"

    for choice in os.listdir(path_handler.path_to_specific_dataset("xy")):
        with open((p := path_handler.path_to_specific_dataset("xy"))
                  + ("/" if p[-1] != "/" else "")
                  + choice, "rb") \
                as file:
            d = pickle.load(file)
            l = d[0] if returnX else d[1]
            for sub in l:
                yield l


def teach(model: RandomForestClassifier, metrics_per_beat: int = 48, n_features=128) -> Embedder:
    """
    Teaches a model

    :param model: the model to be taught
    :param metrics_per_beat: the number of times we compute the FFT per music beat
    :param n_features: the number of frequencies computed by the FFT
    :return: the embedder trained along the model
    """

    embedder, note_count = classify.readyEmbedder(annotations_path)
    print("\033[36;1mStarting to parse database\033[0m to established twinned x and y data")
    file: TextIO

    sacr = [el2.split(".")[0][:-len("_hex")] for el2 in os.listdir(musics_path)]
    choices = [
        el.split(".")[0]
        for el in os.listdir(annotations_path)
        if el.split(".")[0] in sacr
    ]
    del sacr

    for i, choice in enumerate(choices):

        print(f"\rDatabase \033[33;1mtrying to find existing database\033[0m(file \033[36;1m{i}\033[0m /",
              f"\033[34m{len(choices)}\033[0m)",
              end="")
        x: list[list[float]] = []
        y: list[int] = []
        try:
            with open((p := path_handler.path_to_specific_dataset("xy"))
                      + ("/" if p[-1] != "/" else "")
                      + choice + ".pickle", "rb") \
                    as file:

                x, y = pickle.load(file)
                assert isinstance(x, list) and isinstance(y, list) \
                       and isinstance(x[0], list) and isinstance(y[0], int), \
                    f"The data is not correctly stored/extracted. " \
                    f"File {choice}.pickle is corrupted {(x[0][0], y[0])}, " \
                    f"{len(x), len(y)}, {[isinstance(x, list), isinstance(y, list), isinstance(x[0], list), isinstance(y[0], int)]}"
        except (FileNotFoundError, pickle.PickleError, EOFError, AssertionError) as e:

            print(e)

            sound_y, sr = librosa.load(musics_path + (
                "/" if not musics_path.endswith("/") else ""
            ) + choice + "_hex.wav")
            # onset_env = librosa.onset.onset_strength(y=sound_y, sr=sr)
            beat_per_minute = librosa.beat.tempo(y=sound_y, sr=sr)
            tps = beat_per_minute[0] * metrics_per_beat / 60  # metric per second
            jamFile = jams.load(annotations_path + (
                "/" if not annotations_path.endswith("/") else ""
            ) + choice + ".jams")

            hop_len = int(sr / (tps * metrics_per_beat) + .5)
            # tps in beats/s, overSamplingRate in measures/beats, sr in measures/s

            specs = np.abs(librosa.core.stft(sound_y, n_fft=n_features, hop_length=hop_len))
            times = librosa.core.frames_to_time(specs[0], sr=sr, n_fft=n_features, hop_length=hop_len)

            for time_idx, timeStamp in enumerate(times):
                if time_idx % 50 == 0:
                    print(f"\rDatabase \033[33;1mconnecting \033[0m(file \033[36;1m{i}\033[0m /",
                          f"\033[34m{len(choices)}\033[0m",
                          f"TimeIndex : \033[36;1m{time_idx}\033[0m).",
                          end="")
                token = getAnnotation(timeStamp, jamFile)
                try:
                    x.append(list(specs[:, time_idx]))
                    y.append(embedder.map[token][0])
                except KeyError as k:
                    print("\r\033[33;1mCaught", k, " " * 72)

            # storing the data
            with open((p := path_handler.path_to_specific_dataset("xy"))
                      + ("/" if p[-1] != "/" else "")
                      + choice + ".pickle", "wb") \
                    as file:
                pickle.dump((x, y), file)

    model.fit(custom_generator("x"), custom_generator("y"))
    return embedder


if __name__ == '__main__':
    m = build_model()
    embedder = teach(m)
    models_path = path_handler.GNRL_PATH_TO_DATA_SET + "../models/"
    with open(models_path + "RTF.pickle", "wb") as file:
        pickle.dump(m, file)
    del file
    with open(models_path + "EMBED.pickle", "wb") as file:
        pickle.dump(embedder, file)
    del file
