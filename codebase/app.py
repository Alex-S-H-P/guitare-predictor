import os
import pickle
import sys
import typing
import readline
import librosa
import numpy as np

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

from codebase import path_handler
from codebase.utillib import embedder
from sklearn.ensemble import RandomForestClassifier
from codebase.utillib import folder


def load() -> tuple[embedder.Embedder, RandomForestClassifier]:
    """Loads the valid files.

    :returns
        - embedder the embedder that to a note returns a vector, and which can be flipped around
        - model a random forest classifier
    """
    models_path = path_handler.GNRL_PATH_TO_DATA_SET + "../models/"
    with open(models_path + "RTF.pickle", 'rb') as file:
        m = pickle.load(file)
    with open(models_path + "EMBED.pickle", "rb") as file:
        e = pickle.load(file)[0]
    assert isinstance(m, RandomForestClassifier) and isinstance(e, embedder.Embedder)
    return e, m


manualCommands: dict[str, typing.Callable] = {}
MESURES_PAR_BATTEMENT: float = 60
mod: RandomForestClassifier
embed: embedder.Embedder


def main(n_features: int = 128, noteChangeThreshold=75):
    input_str: str = ""
    print("\r\033[37;1m", "-" * 42, "GUITARE-PREDICTOR", "-" * 42, "\033[0m", sep="\n")
    while True:
        try:
            input_str = input(">>> ")
        except EOFError:
            print("\n" + "\033[36;1mBye !\033[0m")
            break
        finally:
            if input_str in manualCommands:
                manualCommands[input_str]()
            elif os.path.isdir(input_str):
                print("File \033[31;1mis a directory\033[0m")
                choices = os.listdir(input_str)
                choices = [f"\033[37;1m{choice}\033[0m/"
                           if os.path.isdir(
                                input_str + (
                                    "/" if input_str[-1] != "/" else ""
                                ) + choice
                           )
                           else f"\033[36;1m{choice}\033[0m" for choice in choices]
                print("Choose among :", *choices, sep="\n\t - ")
            elif os.path.exists(input_str):
                print("File \033[32;1mfound\033[0m")
                sound_y, sr = librosa.load(input_str)
                # onset_env = librosa.onset.onset_strength(y=sound_y, sr=sr)
                tempo = librosa.beat.tempo(y=sound_y, sr=sr)

                hop_len = int(sr / (tempo[0] * MESURES_PAR_BATTEMENT) + .5)
                # tps in beats/s, overSamplingRate in measures/beats, sr in measures/s

                specs = np.abs(librosa.core.stft(sound_y, n_fft=n_features, hop_length=hop_len))
                times = librosa.core.frames_to_time(specs[0], sr=sr, n_fft=n_features, hop_length=hop_len)

                result = [embed.flip(i) + "\n"
                          for i in mod.predict([
                                            list(
                                                specs[:, time_idx]
                                                ) for time_idx, _ in enumerate(times)
                           ])
                          ]
                beat_per_minute = librosa.beat.tempo(y=sound_y, sr=sr)
                result = folder.foldArrayOfNotes(result, beats_per_minute=beat_per_minute,
                                                 metric_per_beats=MESURES_PAR_BATTEMENT,
                                                 noteChangeThreshold=noteChangeThreshold,
                                                 sliding_size=int(beat_per_minute * MESURES_PAR_BATTEMENT / 600 +.5))
                print("Annotations \033[32;1mcréées\033[0m")
                print("Où stocker le résultat ?")
                r = input(">>> ")
                with open(r, "w") as file2:
                    file2.writelines(result)

            else:
                print("File \033[31;1mnot found\033[0m")
                if "/" in input_str:
                    i = max([i for i in range(len(input_str)) if input_str[i] == "/"])
                    if os.path.exists(input_str[:i]):
                        print("Maybe you meant : ")
                        for f in os.listdir(input_str[:i]):
                            print("\t> " +
                                  ("\033[36;1m"
                                   if input_str[i+1:] in f else "\033[37;1m") +
                                  f"{f}\033[0m" +
                                  ("/" if os.path.isdir(input_str[:i] + f)
                                   else "")
                                  )


if __name__ == '__main__':
    if os.getcwd().endswith("codebase") or os.getcwd().endswith("codebase/"):
        os.chdir("..")
    print("\r\033[33mLoading...", end="")
    embed, mod = load()
    main()
