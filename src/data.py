from pathlib import Path

import numpy as np
import torch


class PackedTokens:
    def __init__(self, path, context):
        self.tokens = np.memmap(Path(path), dtype='<u2', mode='r')
        self.context = context
        self.position = 0
        if len(self.tokens) <= context:
            raise ValueError('Shard is too short for one sequence')

    def batch(self, size, device):
        sequences = []
        for _ in range(size):
            if self.position + self.context + 1 > len(self.tokens):
                self.position = 0
            sequences.append(self.tokens[self.position:self.position + self.context + 1])
            self.position += self.context
        tokens = torch.from_numpy(np.asarray(sequences, dtype=np.int64)).to(device)
        return tokens[:, :-1], tokens[:, 1:]
