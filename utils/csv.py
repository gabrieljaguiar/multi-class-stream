from river.stream import iter_pandas, iter_csv
from river.datasets.base import SyntheticDataset
import pandas as pd
from tqdm import tqdm
import os


def save_stream(stream: SyntheticDataset, file: str, size: int):
    # if os.path.exists(file):
    #    return
    if stream is None:
        return
    stream_df_x = []
    stream_df_y = []
    for i, (x, y) in enumerate(stream.take(size)):
        stream_df_x.append(x)
        stream_df_y.append(y)

    stream_df_x = pd.DataFrame(stream_df_x)
    stream_df_y = pd.DataFrame(stream_df_y)

    stream_df = pd.concat([stream_df_x, stream_df_y], axis=1, ignore_index=True)
    stream_df.to_csv(file, index=None)


class CSVStream:
    def __init__(self, csv_file: str, target: str = None, stream_size = None, loop=False) -> None:
        self.csv_file = csv_file
        self.data = pd.read_csv(self.csv_file)
        if target is None:
            self.target = self.data.columns[-1]
        self.classes = self.data[self.target].unique()
        self.n_classes = len(self.classes)
        self.n_features = self.data.shape[1] - 1
        if stream_size == None:
            self.n_samples = self.data.shape[0]
        else:
            self.n_samples = stream_size
        self.index = 0
        self.loop = loop

    def __iter__(self):
        while True:
            row = self.data.iloc[self.index, :-1]
            x = row.to_dict()
            y = self.data.iloc[self.index, -1]
            self.index += 1
            if self.index >= self.n_samples:
                if self.loop:
                    self.index = 0
                else:
                    break
            yield x, y
