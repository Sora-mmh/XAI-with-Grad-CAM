from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from trainer import ClassificationMobileNetV2
import defs
from device import get_device

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Evaluator:
    # Members.
    _data_loaders: Dict[str, DataLoader]
    _trainer: ClassificationMobileNetV2
    _cls: Dict[str, int]
    _eval_pth: Path
    _num_images: int

    def __init__(
        self,
        data_loaders: Dict[str, DataLoader],
        trainer: ClassificationMobileNetV2,
        cls: Dict[str, int],
        eval_pth: Path,
        num_images: int = 6,
    ):
        self._data_loaders = data_loaders
        self._model = trainer.model
        self._cls = cls
        self._eval_pth = eval_pth
        self._num_images = num_images
        self._device = get_device()

    def visualize_predictions(self) -> None:
        training = self._model.training
        self._model.eval()
        image_index = 0
        with torch.no_grad():
            for inputs, labels in self._data_loaders[defs.TEST]:
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)
                outputs = self._model(inputs)
                _, preds = torch.max(outputs, 1)
                for j in range(min(inputs.size()[0], self._num_images)):
                    image_index += 1
                    plt.figure(figsize=(10, 10))
                    ax = plt.subplot(self._num_images // 2, 2, j + 1)
                    ax.axis("off")
                    ax.set_title("predicted: {}".format(self._cls[preds[j]]))
                    img_viewer(inputs.cpu().data[j])
                    if image_index == self._num_images:
                        self._model.train(mode=training)

    def plot_confusion_matrix(
        self, data_type=defs.TEST, file_name="confusion_matrix_test.jpg"
    ) -> None:
        self._model.eval()
        confusion_matrix = torch.zeros(len(self._cls), len(self._cls))
        with torch.no_grad():
            for inputs, labels in self._data_loaders[data_type]:
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)
                outputs = self._model(inputs)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        plt.figure(figsize=(8, 8))
        sns.heatmap(
            confusion_matrix, annot=True, vmin=0, fmt="g", cmap="Blues", cbar=False
        )
        plt.xticks(np.arange(len(self._cls)) + 0.5, self._cls, rotation=90)
        plt.yticks(np.arange(len(self._cls)) + 0.5, self._cls, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig((self._eval_pth / file_name).as_posix())
        plt.show()


def img_viewer(tnsr: torch.Tensor, title: Union[str, None] = None) -> None:
    tnsr = tnsr.numpy().transpose((1, 2, 0))
    mean = np.array(MEAN)
    std = np.array(STD)
    tnsr = std * tnsr + mean
    tnsr = np.clip(tnsr, 0, 1)
    plt.imshow(tnsr)
    if title is not None:
        plt.title(title)
