from river import datasets
from river.datasets import synth
from typing import Dict
import random
import math
from utils.csv import CSVStream


class RealWorldConceptDriftStream(datasets.base.SyntheticDataset):
    def __init__(
        self,
        initialStream_path: str,
        nextStream_path: str,
        classes_affected: list[int],
        width: int,
        position: int,
        size: int,
        angle: float = 0,
        seed: int = 42,
    ):
        self.initialStream = CSVStream(initialStream_path, loop=True)
        self.nextStream = CSVStream(nextStream_path, loop=True)
        self.classes_affected = classes_affected
        self.width = width
        self.position = position
        self.angle = angle
        self.instanceCount = 0
        self.size = size
        self.name = self.initialStream.__class__
        self.seed = seed
        self._rng = random.Random(seed)

    def __str__(self):
        out = "<object, at 0x{}>".format(id(self))
        return out

    def __repr__(self):
        out = "<object, at 0x{}>".format(id(self))
        return out

    def __iter__(self):
        self.initialStreamIterator = iter(self.initialStream)
        self.nextStreamIterator = iter(self.nextStream)
        while True:
            if self.instanceCount == self.size:
                break

            x = -4.0 * (self.instanceCount - self.position) / self.width
            try:
                driftProbability = 1.0 / (1.0 + math.exp(x))
            except:
                driftProbability = 0

            try:
                nextElement = next(self.initialStreamIterator)
                #print (nextElement)
                y_element = nextElement[1]
                if (y_element in self.classes_affected) and (self._rng.random() < driftProbability):
                    nextElement = next(self.nextStreamIterator)
                    while nextElement[1] != y_element:
                        nextElement = next(self.nextStreamIterator)
            except StopIteration:
                break

            self.instanceCount += 1
            yield nextElement

    def reset(self):
        self.instanceCount = 0
        self.initialStreamIterator = iter(self.initialStream)
        self.nextStreamIterator = iter(self.nextStream)
        


