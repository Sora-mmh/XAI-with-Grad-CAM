from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def visualize_saliency_maps(
    saliency_maps: torch.tensor,
    original_image: np.array,
    class_count: int,
    image_width: int = 64,
    image_height: int = 64,
):
    heat_map_colours = [
        (1, 1, 0),
        (0, 1, 1),
    ][:class_count]
    start_colour = (1, 1, 1)
    fused_heatmap_np = None
    for i, saliency_map in enumerate(saliency_maps):
        map_colours = [start_colour, heat_map_colours[i]]
        cmap_tp = LinearSegmentedColormap.from_list("Custom", map_colours, N=256)
        if len(saliency_maps) == 1:
            cmap_tp = plt.get_cmap("jet")
        heatmap_image = Image.fromarray(np.uint8(saliency_map * 255), "L").resize(
            (image_width, image_height), resample=Image.BICUBIC
        )
        heatmap_np = cmap_tp(np.array(heatmap_image))[:, :, :3]
        if i == 0:
            fused_heatmap_np = heatmap_np
        else:
            fused_heatmap_np += heatmap_np
    fused_heatmap_np /= np.max(fused_heatmap_np)
    fused_heatmap = Image.fromarray(np.uint8((fused_heatmap_np * 255)), "RGB")
    original_image = original_image.resize((image_width, image_height))
    new_image = Image.blend(
        original_image.convert("RGB"), fused_heatmap.convert("RGB"), alpha=0.3
    )
    stacked_images = Image.new(
        "RGB", (original_image.width + new_image.width, original_image.height)
    )
    stacked_images.paste(original_image, (0, 0))
    stacked_images.paste(new_image, (original_image.width, 0))
    stacked_images.save(
        "/home/mmhamdi/workspace/classification/XAI-with-fused-multi-class-Grad-CAM/outputs/fmgradcam.jpg"
    )
