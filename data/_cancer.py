import json
import logging
import re
from pathlib import Path
import shutil
from typing import Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


logging.basicConfig(level=logging.INFO)


class CancerDataset(Dataset):

    # Members
    _dataset_pth: Path

    def __init__(self, dataset_pth: Path, data_type: str, transform=None) -> None:
        self._dataset_pth = dataset_pth
        self._data_type = data_type
        self._transform = transform
        self._cls = {
            cls_pth.stem: idx
            for idx, cls_pth in enumerate(
                list((self._dataset_pth / self._data_type).iterdir())
            )
        }
        self._annots_pths = [
            [img_pth, str(cls_pth.stem)]
            for cls_pth in (self._dataset_pth / self._data_type).iterdir()
            for img_pth in cls_pth.iterdir()
        ]

    def __len__(self):
        return len(self._annots_pths)

    def __getitem__(self, index):
        img_pth, label = self._annots_pths[index]
        img_arr = cv2.imread(img_pth.as_posix())
        img_arr = cv2.resize(img_arr, (224, 224))
        img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1)).float()
        if self._transform is not None:
            img_tensor = self._transform(img_tensor)
        return img_tensor, self._cls[label]
