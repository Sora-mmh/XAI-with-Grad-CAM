import gc
import queue
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.nn import functional as F

# sys.path.insert(0, '/content/drive/MyDrive/fm-g-cam')
# from utils._base import visualize_saliency_maps


class FMGradCAM:
    def __init__(
        self,
        model: nn.Module,
        image: torch.tensor,
        target_layer: nn.Conv2d,
        class_count: int,
        activation_fn: torch.nn.functional,
    ):
        self.model = model
        self.image = image
        self.target_layer = target_layer
        self.class_count = class_count
        self.activation_fn = activation_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.queue = queue.Queue()

    def _get_saliency_maps(self):
        self.model.to(self.device)
        self.image.to(self.device)
        self.extract_gradients_with_predictions(plot_pdf=True)
        self.get_weighted_activations()
        self.saliency_maps = torch.cat(self.saliency_maps)
        filtered_maximum_weighted_activations_indexes = self.saliency_maps.argmax(
            dim=0
        ).unsqueeze(0)
        filtered_mask = torch.cat(
            [
                filtered_maximum_weighted_activations_indexes
                for _ in range(self.saliency_maps.size()[0])
            ]
        )
        filtered_mask = torch.cat(
            [
                (
                    filtered_mask[index]
                    == (torch.ones_like(filtered_mask[index]) * index)
                ).unsqueeze(0)
                for index in range(self.saliency_maps.size()[0])
            ]
        ).long()
        self.saliency_maps *= filtered_mask
        self.saliency_maps = F.normalize(self.saliency_maps, p=2, dim=1)
        self.saliency_maps = self.activation_fn(self.saliency_maps)
        self.saliency_maps = self.saliency_maps.detach().cpu().numpy()
        # gc.collect()
        # self.queue.put(self.predictions, self.sorted_predictions_indexes, self.saliency_maps)

    def extract_gradients_with_predictions(
        self, plot_pdf: bool = None, mode: str = None
    ):
        self.gradients_list, self.activations_list = list(), list()
        for p in self.model.parameters():
            p.requires_grad = True
        gradients, activations = None, None

        def hook_backward(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output

        def hook_forward(module, args, output):
            nonlocal activations
            activations = output

        hook_backward = self.target_layer.register_backward_hook(hook_backward)
        hook_forward = self.target_layer.register_forward_hook(hook_forward)
        self.model.eval()
        predictions = self.model(self.image.unsqueeze(0))
        self.sorted_predictions_indexes = torch.argsort(
            predictions, dim=1, descending=True
        ).squeeze(0)
        for c in range(self.class_count):
            predictions[:, self.sorted_predictions_indexes[c]].backward(
                retain_graph=True
            )
            self.gradients_list.append(gradients)
            if plot_pdf is not None:
                kwargs = dict(hist_kws={"alpha": 0.3}, kde_kws={"linewidth": 2})
                colors = sns.color_palette("Paired")[:12]
                averaged_gradients = torch.mean(gradients[0], dim=1)
                plt.figure(figsize=(10, 7), dpi=80)
                if mode == "relu":
                    gradients = torch.nn.functional.relu(gradients)
                    averaged_gradients = torch.nn.functional.relu(averaged_gradients)
                if mode == "gelu":
                    gradients = torch.nn.functional.gelu(gradients)
                    averaged_gradients = torch.nn.functional.gelu(averaged_gradients)
                if mode == "sigmoid":
                    gradients = torch.nn.functional.sigmoid(gradients)
                    averaged_gradients = torch.nn.functional.sigmoid(averaged_gradients)

                sns.distplot(
                    averaged_gradients.detach().cpu().numpy().squeeze(),
                    color="red",
                    label=f"averaged gradients - {mode}",
                    **kwargs,
                )
                plt.legend()
                for channel in range(gradients[0].size()[1]):
                    gradients_per_channel = gradients[0][
                        :,
                        channel,
                        :,
                    ]
                    sns.distplot(
                        gradients_per_channel.detach().cpu().numpy().squeeze(),
                        color=colors[channel],
                        label=f"channel {channel + 1} - {mode}",
                        **kwargs,
                    )
                    plt.legend()
                    plt.savefig(
                        f"C:/Users/monta/Documents/MONT@/FM-G-CAM/fm-g-cam/gradients/pdf-channel-{channel}.png"
                    )

            self.activations_list.append(activations)
        hook_backward.remove()
        hook_forward.remove()
        for p in self.model.parameters():
            p.requires_grad = False
        self.predictions = predictions.squeeze().detach().cpu().numpy()

    def get_weighted_activations(self):
        self.saliency_maps = list()
        for activations, gradients in zip(self.activations_list, self.gradients_list):
            averaged_pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
            for idx in range(activations.size()[1]):
                activations[:, idx, :, :] *= averaged_pooled_gradients[idx]
            saliency_map = torch.mean(activations, dim=1).squeeze()
            self.saliency_maps.append(saliency_map.unsqueeze(0).detach().cpu())
