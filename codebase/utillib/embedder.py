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
