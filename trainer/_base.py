from abc import abstractmethod, ABC
import copy
import logging
from pathlib import Path
import time
from typing import Dict
from matplotlib import pyplot as plt

import torch
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim

import defs
from device import get_device


logging.basicConfig(level=logging.INFO)


class ClassificationModel(ABC):

    # Members.
    _data_loaders: Dict[str, DataLoader]
    _data_sizes: Dict[str, int]
    _cls: Dict[str, int]
    _model: torch.nn.Module
    _epochs: int
    _freeze: int

    def __init__(
        self,
        data_loaders: Dict[str, DataLoader],
        data_sizes: Dict[str, int],
        cls: Dict[str, int],
        epochs: int,
        freeze: int,
    ):
        self._data_loaders = data_loaders
        self._data_sizes = data_sizes
        self._cls = cls
        self._epochs = epochs
        self._freeze = freeze
        self._device = get_device()
        self._build_model()

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def data_loaders(self) -> Dict[str, DataLoader]:
        return self._data_loaders

    @property
    def cls(self) -> Dict[str, int]:
        return self._cls

    @abstractmethod
    def _build_model(self) -> None:
        return NotImplemented

    @abstractmethod
    def train_model(self) -> None:
        return NotImplemented

    @abstractmethod
    def _show_metrics(self) -> None:
        return NotImplemented


class ClassificationMobileNetV2(ClassificationModel):
    def _build_model(self) -> None:
        self._model = models.mobilenet_v2(pretrained=True)
        if self._freeze is not None:
            l_count = 0
            for layer in list(self._model.children())[0]:  # type: ignore
                l_count += 1
                if l_count < self._freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
        num_features = self._model.classifier[1].in_features
        self._model.classifier[1] = torch.nn.Linear(num_features, len(self._cls))  # type: ignore
        self._model = self._model.to(self._device)
        self._criterion = torch.nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=1e-4
        )
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, step_size=7, gamma=0.1
        )

    def train_model(self) -> None:
        start = time.time()
        best_model_wts = copy.deepcopy(self._model.state_dict())
        best_acc = 0.0
        losses, accs = [], []
        for epoch in range(self._epochs):
            logging.info(f"Epoch {epoch+1}/{self._epochs}")
            for phase in [defs.TRAIN, defs.VAL]:
                if phase == defs.TRAIN:
                    self._model.train()
                else:
                    self._model.eval()
                running_loss = 0.0
                running_corrects = 0
                data_loader = self._data_loaders[phase]
                for inputs, labels in data_loader:
                    inputs = inputs.to(self._device)
                    labels = labels.to(self._device)
                    self._optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == defs.TRAIN):
                        outputs = self._model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self._criterion(outputs, labels)
                        if phase == defs.TRAIN:
                            loss.backward()
                            self._optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == defs.TRAIN:
                    self._scheduler.step()
                epoch_loss = running_loss / self._data_sizes[phase]
                losses.append(epoch_loss)
                epoch_acc = running_corrects.double() / self._data_sizes[phase]  # type: ignore
                accs.append(epoch_acc.item())
                if phase == defs.VAL and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self._model.state_dict())
                    torch.save(
                        best_model_wts,
                        Path(defs.WEIGHTS_DIR_PTH) / "MobileNetV2_best_wts.pth",
                    )
                logging.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                if phase == defs.VAL and epoch_acc > best_acc:
                    best_acc = epoch_acc
        training_duration = time.time() - start
        logging.info(
            f"Training complete in {training_duration // 60:.0f}m {training_duration % 60:.0f}s"
        )
        logging.info(f"Best val Acc: {best_acc:4f}")
        self._train_loss = [e for idx, e in enumerate(losses) if idx % 2 == 0]
        self._val_loss = [e for idx, e in enumerate(losses) if idx % 2 != 0]
        self._train_acc = [e for idx, e in enumerate(accs) if idx % 2 == 0]
        self._val_acc = [e for idx, e in enumerate(accs) if idx % 2 != 0]
        self._show_metrics()

    def _show_metrics(self) -> None:
        plt.style.use("fivethirtyeight")
        _, (x_axis, y_axis) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))  # type: ignore
        x_axis.plot(range(self._epochs), self._train_loss, "r", label="Training loss")
        x_axis.plot(range(self._epochs), self._val_loss, "g", label="Validation loss")
        x_axis.set_title("Training and Validation Loss")
        x_axis.set_xlabel("Epochs")
        x_axis.set_ylabel("Loss")
        x_axis.legend()
        y_axis.plot(
            range(self._epochs), self._train_acc, "r", label="Training Accuracy"
        )
        y_axis.plot(
            range(self._epochs), self._val_acc, "g", label="Validation Accuracy"
        )
        y_axis.set_title("Training and Validation Accuracy")
        y_axis.set_xlabel("Epochs")
        y_axis.set_ylabel("Accuracy")
        y_axis.legend()
        plt.tight_layout
        plt.savefig(
            "/home/mmhamdi/workspace/classification/XAI-with-fused-multi-class-Grad-CAM/outputs/losses.jpg"
        )
        # plt.show()
