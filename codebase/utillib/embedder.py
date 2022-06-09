"""
The embedder, stored in a single file for comprehensibility.

Author : Alexandre SCHŒPP https://github.com/Alex-S-H-P/
"""

import pickle

import numpy as np


class Embedder:
    """
    Assigns to a finite set of tokens id according to their order of discovery

    Can create a vector which is zero except for the coordinate corresponding to the index of a token.
    Can flip back an index into its corresponding token
    Can inverse a vector into its corresponding token
    """

    def __init__(self):
        # Map : token -> (index, counter)
        self.map: dict[str, tuple[int, int]] = {}

    def add_value(self, value: str):
        """
        :param value: The new value in the embedder
        """
        if value in self.map:
            val = self.map[value]
            self.map[value] = val[0], val[1] + 1
        else:
            self.map[value] = len(self.map), 1

    def vectorialize(self, token: str) -> np.ndarray:
        """
        :param token: A token in the set
        :return: a vector with coordinate 1 at the token's index and 0 everywhere else.
        """
        if token not in self.map:
            raise KeyError(f"{token} has never been recorded before")
        val = np.zeros(len(self.map), dtype="float64")
        val[self.map[token][0]] = 1.
        return val

    def reverse(self, item: np.ndarray):
        """
        :param item: A vector
        :return: the token corresponding to it's max.
        """

        i = np.argmax(item)
        for k in self.map:
            if self.map[k][0] == i:
                return k
        raise KeyError(f"Index {i} found. Could hasn't generated an embedding for this key")

    def flip(self, i: int) -> str:
        """
        :param i: An index
        :return: its corresponding token
        """
        for k in self.map:
            if self.map[k][0] == i:
                return k
        raise KeyError(f"Index {i} found. Could hasn't generated an embedding for this key")

    def __len__(self) -> int:
        """
        :return: The number of tokens discovered, which is also the dimension of any returned vector
        """
        return len(self.map)

    def save(self, path: str):
        """Saves the token"""
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
