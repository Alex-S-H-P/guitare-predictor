import os
import pickle
import typing

import librosa
import numpy as np

try:
    from codebase import path_handler
    from codebase.utillib import embedder
except ModuleNotFoundError:
    import path_handler
    from utillib import embedder
finally:
    from sklearn.ensemble import RandomForestClassifier


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


def main(n_features: int = 128):
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
            if os.path.exists(input_str):
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
                        for f in os.listdir():
                            print(f"\t> {f}")


if __name__ == '__main__':
    if os.getcwd().endswith("codebase") or os.getcwd().endswith("codebase/"):
        os.chdir("..")
    print("\r\033[33mLoading...", end="")
    embed, mod = load()
    main()
