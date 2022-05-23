import pickle

import numpy as np


class Embedder:

    def __init__(self):
        """Map : token -> (index, counter)"""
        self.map: dict[str, tuple[int, int]] = {}

    def add_value(self, value: str):
        if value in self.map:
            val = self.map[value]
            self.map[value] = val[0], val[1] + 1
        else:
            self.map[value] = len(self.map), 1

    def vectorialize(self, token: str) -> np.ndarray:
        if token not in self.map:
            raise KeyError(f"{token} has never been recorded before")
        val = np.zeros(len(self.map), dtype="float64")
        val[self.map[token][0]] = 1
        return val

    def reverse(self, item: np.ndarray):
        i = np.argmax(item)
        for k in self.map:
            if self.map[k][0] == i:
                return k
        raise KeyError(f"Index {i} found. Could hasn't generated an embedding for this key")

    def __len__(self) -> int:
        return len(self.map)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.map, f)

    @classmethod
    def load(cls, path: str):
        """
        Loads the dictionary at %path, and builds an embedder around it.
        """
        with open(path, "rb") as f:
            m = pickle.load(f)
        instance = cls()
        instance.map = m
        return instance
