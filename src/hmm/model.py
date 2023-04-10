from collections import Counter
from dataclasses import dataclass
from itertools import chain
import numpy as np


@dataclass
class HMM:
    """
    A: (s, s') -> probability. s is current state, s' is next state
    B: (s, x) -> probability. s is current state, x is observation
    """
    O: set[str]
    S: set[str]

    def __post_init__(self):
        K = len(self.S)
        N = len(self.O)

        self.PI = np.zeros(shape=K, dtype=np.float64)
        self.A = np.zeros(shape=(K, K), dtype=np.float64)
        self.B = np.zeros(shape=(K, N), dtype=np.float64)
        
        self.X_index: dict[str, int] = {x: i for i, x in enumerate(self.O)}
        self.Y_index: dict[str, int] = {y: i for i, y in enumerate(self.S)}
        self.index_to_Y = [key for key, value in self.Y_index.items()]

    def train(self, X: list[list[str]], Y: list[list[str]]) -> None:

        N = len(Y)

        counts = Counter(chain(*Y))

        emission_temp = [list(zip(y, x)) for (y, x) in zip(Y, X)]

        emission_counts = Counter(chain(*emission_temp))

        prior_counts = Counter([sequence[0] for sequence in Y])

        bigrams = [list(zip(y, y[1:])) for y in Y]
        
        transition_counts = Counter(chain(*bigrams))

        # Prior Probabilities
        for s in self.S:
            self.PI[self.Y_index[s]] = prior_counts[s] / N
        
        # Transition probabilities
        for s in self.S:
            for s_ in self.S:
                self.A[self.Y_index[s], self.Y_index[s_]] = transition_counts[s, s_] / counts[s]
        
        # Emission probabilities
        for o in self.O:
            for s in self.S:
                self.B[self.Y_index[s], self.X_index[o]] = emission_counts[s, o] / counts[s]




