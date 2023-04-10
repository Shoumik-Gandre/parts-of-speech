from dataclasses import dataclass
import numpy as np
from ..model import HMM


@dataclass
class ViterbiDecoder:

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
        N = len(self.hmm.O)
        K = len(self.hmm.S)

        T1 = np.zeros((K, T), dtype=np.float64)
        T2 = np.zeros((K, T), dtype=int)
        
        # Remap input sequence to an array of integer indexes
        Xi = np.array([X_index[x] for x in X], dtype=int) # shape = T
        Yi = np.zeros(T, dtype=np.int32)

        T1[..., 0] = PI * B[..., Xi[0]]
        
        for t in range(1, T):
            M = T1[..., t-1].reshape(-1, 1) * A * B[..., Xi[t]]
            T1[..., t] = np.max(M, axis=0)
            T2[..., t] = np.argmax(M, axis=0)
        
        Yi[T-1] = np.argmax(T1[..., T-1])

        for t in reversed(range(1, T)):
            Yi[t-1] = T2[Yi[t], t]
        
        Y = [self.hmm.index_to_Y[Yi[t]] for t in range(T)]
        
        return Y


@dataclass
class ViterbiDecoderLog:

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

        X_index = self.hmm.X_index

        T = len(X)
        N = len(self.hmm.O)
        K = len(self.hmm.S)

        T1 = np.zeros((K, T), dtype=np.float64)
        T2 = np.zeros((K, T), dtype=int)
        
        # Remap input sequence to an array of integer indexes
        Xi = np.array([X_index[x] for x in X], dtype=int) # shape = T
        Yi = np.zeros(T, dtype=int)

        T1[..., 0] = np.log(PI * B[..., Xi[0]] + np.finfo(float).eps)
        
        for t in range(1, T):
            M = T1[..., t-1].reshape(-1, 1) + np.log(A + np.finfo(float).eps) + np.log(B[..., Xi[t]] + np.finfo(float).eps)
            T1[..., t] = np.max(M, axis=0)
            T2[..., t] = np.argmax(M, axis=0)
        
        Yi[T-1] = np.argmax(T1[..., T-1])

        for t in reversed(range(1, T)):
            Yi[t-1] = T2[Yi[t], t]
        
        Y = [self.hmm.index_to_Y[Yi[t]] for t in range(T)]
        
        return Y