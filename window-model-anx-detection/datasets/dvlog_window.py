from pathlib import Path
from typing import Union
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class DVlogWindow(data.Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        fold="train",
        gender="both",
        window_size=10,          
        overlap=0.5,             
    ):

        self.root = root if isinstance(root, Path) else Path(root)
        self.fold = fold
        self.gender = gender

        self.window_size = window_size
        self.step = int(window_size * (1 - overlap))   # sliding step

        self.features = []
        self.labels = []

        with open(self.root / "labels.csv", "r") as f:

            for line in f:
                sample = line.strip().split(",")

                if not self.is_sample(sample):
                    continue

                s_id = sample[0]

                label = int(sample[1] == "anxious")  # same as original

                v_feature = np.load(self.root / s_id / f"{s_id}_visual.npy")
                a_feature = np.load(self.root / s_id / f"{s_id}_acoustic.npy")

                T_v, T_a = v_feature.shape[0], a_feature.shape[0]
                T = min(T_v, T_a)

                v_feature = v_feature[:T]
                a_feature = a_feature[:T]

                feature = np.concatenate((v_feature, a_feature), axis=1).astype(
                    np.float32
                )

                # =========================
                #  WINDOWING IMPLEMENTATION
                # =========================
                for start in range(0, T - window_size + 1, self.step):

                    end = start + window_size

                    window = feature[start:end]

                    if window.shape[0] == window_size:
                        self.features.append(window)
                        self.labels.append(label)

    def is_sample(self, sample):

        gender, fold = sample[3], sample[4]

        if self.gender == "both":
            return fold == self.fold

        return (fold == self.fold) and (gender == self.gender)

    def __getitem__(self, i):

        feature = self.features[i]
        label = self.labels[i]                      
        return feature, label                               #returns window feartures and labels

    def __len__(self):

        return len(self.labels)


def _collate_fn(batch):

    features, labels = zip(*batch)

    padded_features = pad_sequence(
        [torch.from_numpy(f) for f in features], batch_first=True       #ctually no padding happens
    )

    labels = torch.tensor(labels)

    return padded_features, labels


def get_dvlog_window_dataloader(
    root,
    fold="train",
    batch_size=8,
    gender="both",
    window_size=10,
    overlap=0.5,
):

    dataset = DVlogWindow(
        root=root,
        fold=fold,
        gender=gender,
        window_size=window_size,
        overlap=overlap,
    )

    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(fold == "train"),
        collate_fn=_collate_fn,
    )

    return dataloader