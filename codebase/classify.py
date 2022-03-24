import json
import os
import typing

import jams
import matplotlib.pyplot as plt
import numpy
import numpy as np
import sklearn.cluster as cluster

import codebase.utillib.embedder as embed

from codebase import path_handler


def getData(path: str) -> np.ndarray:
    """Returns the data as a numpy array that is contained within the guitar database"""
    choices = os.listdir(path)
    embedder = embed.Embedder()
    print("\033[36;1mStarting to parse database\033[0m to evaluate embedding size")
    for i, choice in enumerate(choices):
        print(f"\rEmbedder \033[33;1mcomputing\033[0m (\033[36m{i}\033[0m / \033[36m{len(choices)}\033[0m).", end="")
        jam = jams.load(path + ("/" if not path.endswith("/") else "") + choice)
        for instructed_cords in jam.search(namespace="chord"):
            chord = json.loads(instructed_cords.__str__())
            data = chord["data"]
            for sample in data:
                embedder.add_value(sample["value"])
    print("\rEmbedder \033[32;1mready\033[0m", " "*64)
    print("\033[36;1mCreating data vector\033[0m")
    mega_vector = np.zeros((1, len(embedder.map)), dtype="float64")
    for i, choice in enumerate(choices):
        print(f"\r\033[33;1mReading\033[0m (\033[36m{i}\033[0m / \033[36m{len(choices)}\033[0m).", end="")
        jam = jams.load(path + ("/" if not path.endswith("/") else "") + choice)
        for instructed_cords in jam.search(namespace="chord"):
            chord = json.loads(instructed_cords.__str__())
            data = chord["data"]
            for sample in data:
                vector = embedder.vectorialize(sample["value"])
                mega_vector = np.concatenate((mega_vector, vector.reshape(1, -1)), 0)
    print("\rData \033[32;1mread\033[0m", " " * 64)
    mega_vector = mega_vector[1:, :]
    print(f"Found \033[36;1m {mega_vector.shape[0]} notes played, including {mega_vector.shape[1]} different")
    return mega_vector


def determine_the_amount_of_classes(k_range: typing.Iterable, data: np.ndarray) -> None:
    """Displays the inertia curve along k values in k_range to choose k optimally"""
    K: list[int] = [k for k in k_range]
    I: list[float] = []
    for k in k_range:
        model = cluster.KMeans()
        model.fit(data)
        I.append(model.inertia_)
    plt.plot(K, I)


if __name__ == '__main__':
    getData(path_handler.path_to_specific_dataset("annotations"))
