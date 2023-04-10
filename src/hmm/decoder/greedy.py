from dataclasses import dataclass

import numpy as np

from ..model import HMM


@dataclass
class GreedyDecoder:

    hmm: HMM

    def decode(self, X: list[str]) -> list[str]:
        """
        O: observation space
        S: state space
        PI: initial probabilities
        X: sequence of observations
        A: transition matrix
        B: emission matrix
        """
        A = self.hmm.A
        B = self.hmm.B
        PI = self.hmm.PI

        Y_index = self.hmm.Y_index
        X_index = self.hmm.X_index

        T = len(X)
        
        # Remap input sequence to an array of integer indexes
        Xi = np.array([X_index[x] for x in X], dtype=int) # shape = T
        Yi = np.zeros(T, dtype=int)

        state = np.argmax(PI * B[:, Xi[0]])
        Yi[0] = state

        for t in range(1, T):
            state = np.argmax(A[state, :] * B[:, Xi[t]])
            Yi[t] = state

        Y = [self.hmm.index_to_Y[Yi[t]] for t in range(T)]

        return Y