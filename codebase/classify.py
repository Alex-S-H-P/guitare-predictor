"""
General functions useful for classifying notes.

When ran as a standalone, does the elbow method to display the usefulness of each class.

Author : Alexandre SCHŒPP https://github.com/Alex-S-H-P/
"""

import json
import os
import typing

import jams
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as cluster

from codebase.utillib import embedder as embed, minMaxContext as normalizer
from codebase.utillib.annotations import on_fail_ask_user_politely

from codebase import path_handler


def readyEmbedder(path, verbose: bool = True, norm=normalizer.Normalizer(), jam=None) -> tuple[embed.Embedder, int]:
    choices = os.listdir(path)
    embedder = embed.Embedder()
    note_count = 0
    if verbose:
        print("\033[36;1mStarting to parse database\033[0m to evaluate embedding size")

    # first pass for embedding and normalizing pre-computing.
    for i, choice in enumerate(choices):
        if verbose:
            # print which file is being read.
            print(f"\rEmbedder \033[33;1mcomputing\033[0m (file \033[36m{i}\033[0m / "
                  f"\033[36m{len(choices)}\033[0m, found \033[34;1m{len(embedder)}\033[0m classes, "
                  f"read \033[34;1m{note_count}\033[0m).",
                  end="")

        jam = jams.load(path + ("/" if not path.endswith("/") else "") + choice)
        data = jam.search(namespace="chord")[1]["data"]
        for sample in data:
            note_count += 1
            embedder.add_value(sample.value)
            norm.fit_add(sample.duration)
    if verbose:
        print(f"\rEmbedder \033[32;1mDONE\033[0m "
              f"(\033[36;1m{note_count}\033[0m notes found, \033[36;1m{len(embedder)}\033[0m classes)")
    return embedder, note_count


@on_fail_ask_user_politely
def getData(path: str, norm=normalizer.Normalizer(), embedder:embed.Embedder=None, note_count:int = None) -> np.ndarray:
    """
    Returns the data as a numpy array that is contained within the guitar database

    We use normalizer.Normalizer() as a default value for the norm key word because
    we want it to be shared across executions by default.
    """
    choices = os.listdir(path)
    if embedder is None or note_count is None:
        embedder = embed.Embedder()
        note_count = 0
        print("\033[36;1mStarting to parse database\033[0m to evaluate embedding size")

        # first pass for embedding and normalizing pre-computing.
        for i, choice in enumerate(choices):

            # print which file is being read.
            print(f"\rEmbedder \033[33;1mcomputing\033[0m (file \033[36m{i}\033[0m / "
                  f"\033[36m{len(choices)}\033[0m, found \033[34;1m{len(embedder)}\033[0m classes, "
                  f"read \033[34;1m{note_count}\033[0m).",
                  end="")

            jam = jams.load(path + ("/" if not path.endswith("/") else "") + choice)
            for instructed_cords in jam.search(namespace="chord"):
                chord = json.loads(instructed_cords.__str__())
                data = chord["data"]
                for sample in data:
                    note_count += 1
                    embedder.add_value(sample["value"])
                    norm.fit_add(sample["duration"])

        print("\rEmbedder \033[32;1mready\033[0m", " " * 64)
    else:
        print("\rEmbedder \033[32;1mFound\033[0m", " " * 64)
    print(f"\033[36;1mCreating data vector for the \033[34;1m{note_count}\033[36;1m notes found \033[0m")

    # we make the vector
    mega_vector = np.zeros((note_count, len(embedder.map) + 1), dtype="float64")
    k = 0

    # second pass to actually populate this database vector.
    for i, choice in enumerate(choices):
        # print which file is being read.
        print(f"\r\033[33;1mReading\033[0m (\033[36m{i}\033[0m / \033[36m{len(choices)}\033[0m).", end="")
        jam = jams.load(path + ("/" if not path.endswith("/") else "") + choice)
        for instructed_cords in jam.search(namespace="chord"):
            chord = json.loads(instructed_cords.__str__())
            data = chord["data"]
            for sample in data:
                normalized_meta_data = np.array(
                    [
                        norm(sample["duration"])
                    ]
                )

                mega_vector[k] = np.concatenate((
                    embedder.vectorialize(sample["value"]),
                    normalized_meta_data)
                )
                k += 1
    # Data is read. tell the user.
    print("\rData \033[32;1mread\033[0m", " " * 64)
    print(
        f"Found \033[36;1m {mega_vector.shape[0]}\033[0m notes played, including \033[36;1m{mega_vector.shape[1] - 1}\033[0m different")
    # returns.
    norm = np.linalg.norm(mega_vector)
    print("Norm of database vector :",
          "\033[32;1m" if norm > 1 else "\033[31;1m",
          norm,
          "\033[0m")
    return mega_vector


def determine_the_amount_of_classes(k_range: typing.Iterable, data: np.ndarray) -> None:
    """Displays the inertia curve along k values in k_range to choose k optimally"""
    K: list[int] = [k for k in k_range]
    I: list[float] = []
    print("\033[36;1mTesting appropriate number of classes.\033[0m")
    for k in k_range:
        print(f"\r\033[33;1mTesting \033[34;1mk = {k}\033[0m", end="")
        model = cluster.KMeans(n_clusters=k, n_init=20)
        model.fit(data)
        I.append(model.inertia_)
    print(f"\r\033[33;1mTesting \033[32;1mCOMPLETE\033[0m")
    plt.plot(K, I)
    plt.show()


if __name__ == '__main__':
    vector = getData(path_handler.path_to_specific_dataset("annotations"))
    determine_the_amount_of_classes([10, 30, 63, 100, 150, 200, 300, 400, 500, 630, 631], vector)
