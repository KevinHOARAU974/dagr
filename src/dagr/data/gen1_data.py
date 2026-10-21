import torch
import numpy as np

from torch.utils.data import Dataset
from torch_geometric.data import Data

from prophesee_toolbox.io.dat_events_tools import load_td_data

from dagr.data.augment import init_transforms
from dagr.data.utils import to_data

from pathlib import Path
from typing import Optional, Callable


class Gen1(Dataset):

    def __init__(self, root: Path, split, transform=Optional[Callable[[Data,], Data]], num_events: int=50000):
        super().__init__()

        self.load_dir = root / split
        self.classes = ["car", "pedestrian"]
        self.num_classes = 2

        with open(self.load_dir / "dataset_information.txt", "r") as f:
            samples_txt = f.readlines()

        self.samples_idx = []
        for sample in samples_txt:

            sample_plit = sample.split(" ")
            self.samples_idx.append((sample_plit[0], int(sample_plit[1]), int(sample_plit[2])))

        dtype = np.dtype([
            ("file_name", "U100"),
            ("ts", np.uint64),
            ("high", np.int64)
        ])

        self.samples_idx = np.array(self.samples_idx, dtype=dtype)

        self.height = 240
        self.width = 304

        if transform is not None and hasattr(transform, "transforms"):
            init_transforms(transform.transforms, self.height, self.width)
        self.transform = transform
        self.time_window = 1000000
        self.num_events = num_events

    def __len__(self):
        return len(self.samples_idx)

    def preprocess(self, data):
        data.t -= (data.t[-1] - self.time_window + 1)
        return data

    def __getitem__(self, index):

        file_name, ts, high =  self.samples_idx[index]

        low = max(0, high-self.num_events)

        file_dat = file_name.replace("bbox.npy", "td.dat")

        events = self._load_event(file_dat, low, high)

        data = to_data(**events, bbox=self.load_bboxes(file_name, ts),
                t0=events['t'][0], t1=events['t'],
                width=self.width, height=self.height,
                time_window=self.time_window)

        data = self.preprocess(data)

        data = self.transform(data) if self.transform is not None else data
        
        if not hasattr(data, "t"):
            data.t = data.pos[:, -1:]
            data.pos = data.pos[:, :2].type(torch.int16)

        return data

    def _load_event(self, f_path, low, high):

        events  = load_td_data(self.load_dir / f_path, ev_start=low, ev_count=high - low)

        x=events['x']
        y=events['y']
        t=events['t']
        p=events['p']

        return dict(x=x, y=y, t=t, p=p)

    def load_bboxes(self, file, timestamp):

        bboxes = np.load(self.load_dir / file)
        bboxes = bboxes[bboxes['ts'] == timestamp]

        return np.stack((
            bboxes['x'],
            bboxes['y'],
            bboxes['w'],
            bboxes['h'],
            bboxes['class_id'],
            bboxes['confidence']
        ), axis=1).astype(np.float32)
        