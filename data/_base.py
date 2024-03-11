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


class DatasetGenerator:

    # Members
    _data_pth: Path
    _root_dir: Path

    def __init__(self, data_pth: Path, root_dir: Path) -> None:
        self._data_pth = data_pth
        self._root_dir = root_dir
        self._metadata_pth = data_pth / "meta.json"
        self._annots_pth = data_pth / "File1" / "ann"
        self._imgs_pth = data_pth / "File1" / "img"
        logging.info("Start extracting classes and rectifying their indices ...")
        self.get_cls_with_indices_from_metadata()
        logging.info("Extraction and rectification are completed.")
        logging.info("Start cropping and saving annotations ...")
        self.save_cropped_annots()
        logging.info("Cropping annotations completed.")
        logging.info("Start splitting data ...")
        self.split_data()
        logging.info("Splitting data is completed.")

    def get_cls_with_indices_from_metadata(self):
        with open(self._metadata_pth) as f:
            classes = json.load(f)
        cls_indices = {
            cls["title"]: (cls["id"], cls["color"]) for cls in classes["classes"]
        }
        self._cls_indices = {k: i for i, (k, _) in enumerate(cls_indices.items())}

    def save_cropped_annots(self):
        for pth in self._annots_pth.iterdir():
            img_arr = cv2.imread((self._imgs_pth / pth.stem).as_posix())
            if img_arr is not None:
                with open(pth) as f:
                    objects = json.load(f)["objects"]
                for obj in objects:
                    x_min, y_min, x_max, y_max = convert_polygon_to_bbox(
                        obj["points"]["exterior"]
                    )
                    cropped_annot = img_arr[y_min:y_max, x_min:x_max]
                    cropped_annot_pth = (
                        root_dir
                        / str(
                            "img"
                            + "/"
                            + re.findall("\d+", pth.stem)[0]
                            + "_"
                            + obj["classTitle"]
                            + ".jpg"
                        )
                    ).as_posix()
                    cv2.imwrite(cropped_annot_pth, cropped_annot)

    def split_data(self, split_ratio: tuple = (0.7, 0.2, 0.1)):
        all_annots_pths = [
            annot_pth
            for annot_pth in (root_dir / "img").iterdir()
            if annot_pth.as_posix().endswith(".jpg")
        ]
        annots_pths_dict = {
            "train": all_annots_pths[: int(len(all_annots_pths) * split_ratio[0])],
            "val": all_annots_pths[
                int(len(all_annots_pths) * split_ratio[0]) : int(
                    len(all_annots_pths) * (1 - split_ratio[2])
                )
            ],
            "test": all_annots_pths[int(len(all_annots_pths) * (1 - split_ratio[2])) :],
        }
        for data_type, imgs in annots_pths_dict.items():
            (root_dir / data_type).mkdir(exist_ok=True)
            with open((root_dir / (data_type + ".txt")).as_posix(), "w") as f:
                for img in imgs:
                    shutil.copy(img, (root_dir / data_type / img.name).as_posix())
                    f.write((root_dir / data_type / img.name).as_posix() + "\n")


class DamagesDataset(Dataset):

    # Members
    _dataset_pth: Path

    def __init__(
        self, dataset_pth: Path, data_type: str, cls: Dict[str, int], transform=None
    ) -> None:
        self._dataset_pth = dataset_pth
        self._data_type = data_type
        self._cls = cls
        self._transform = transform
        self._annots_pths = list((self._dataset_pth / data_type).iterdir())

    def __len__(self):
        return len(self._annots_pths)

    def __getitem__(self, index):
        annot_pth = self._annots_pths[index]
        annot_label = annot_pth.stem.split("_")[-1]
        img_arr = cv2.imread(annot_pth.as_posix())
        img_arr = cv2.resize(img_arr, (64, 64))
        img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1)).float()
        if self._transform is not None:
            img_tensor = self._transform(img_tensor)
        return img_tensor, self._cls[annot_label]


def convert_polygon_to_bbox(polygon):
    polygon_arr = np.array(polygon)
    x_min, x_max = polygon_arr[:, 0].min(), polygon_arr[:, 0].max()
    y_min, y_max = polygon_arr[:, 1].min(), polygon_arr[:, 1].max()
    return int(x_min), int(y_min), int(x_max), int(y_max)


if __name__ == "__main__":
    data_pth = Path(
        "/home/mmhamdi/workspace/classification/dataset/car_damages_dataset"
    )
    root_dir = Path(
        "/home/mmhamdi/workspace/classification/dataset/car_damages_dataset/data"
    )

    data_formatter = DatasetGenerator(data_pth=data_pth, root_dir=root_dir)
