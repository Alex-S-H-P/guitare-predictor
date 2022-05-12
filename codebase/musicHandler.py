import os
import pickle

import jams
import librosa
import path_handler
import classify
from sklearn.ensemble import RandomForestClassifier

annotations_path = path_handler.path_to_specific_dataset("annotations")
musics_path = path_handler.path_to_specific_dataset("music")


def build_model() -> RandomForestClassifier:
    """Constructs a model"""
    model = RandomForestClassifier(n_jobs=-1)
    return model


def teach(model: RandomForestClassifier, how_many: str | int, overSamplingRate: int = 48):
    """teaches a model"""
    assert isinstance(how_many, int) or how_many.strip(' -\t\n')
    print("\033[36;1mStarting to parse database\033[0m to established twinned x and y data")
    embedder, note_count = classify.readyEmbedder(annotations_path)
    x :list [list[float]] = []
    y : list[list[float]] = []

    choices = [
        el
        for el in os.listdir(annotations_path)
        if el not in os.listdir(musics_path)
    ]
    for i, choice in enumerate(choices):
        print(f"\rDatabase \033[33;1mwriting\033[0m (file \033[36m{i}\033[0m / {len(choices)}).",
              end="")
        sound_y, sr = librosa.load(musics_path + (
                "/" if not musics_path.endswith("/") else ""
            ) + choice)
        onset_env = librosa.onset.onset_strength(y=sound_y, sr=sr)
        tempo = librosa.beat.tempo(sound_y, sr=sr)
        jams.load(annotations_path + (
                "/" if not annotations_path.endswith("/") else ""
            ) + choice)
        librosa.core.fft_frequencies(128)
        librosa.core.frames_to_time(sr, tempo*overSamplingRate, )


    m.fit(x, y)


if __name__ == '__main__':
    m = build_model()
    teach(m, "full data-set")
    models_path = path_handler.GNRL_PATH_TO_DATA_SET + "../models/"
    file = open(models_path + "RTF.pickle", "w")
    pickle.dump(m, file)
