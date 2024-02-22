import logging

logging.basicConfig(level=logging.INFO)
import requests
import sys
import threading

from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F

from trainer._base import Net
from fmgcam._base import FMGradCAM
from utils._base import visualize_saliency_maps


if __name__ == "__main__":
    # sys.path.insert(0, "/fm-g-cam")
    model = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(
        torch.load(
            "C:/Users/monta/Documents/MONT@/FM-G-CAM/fm-g-cam/checkpoints/cifar_net.pth"
        )
    )
    target_layer = model.conv2
    class_count = 4
    activation_fn = F.relu
    image_width, image_height = 32, 32

    url = "https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/input_images/cat_dog.png"
    r = requests.get(url, allow_redirects=True)
    open("dog-and-cat-cover.jpg", "wb").write(r.content)
    image = Image.open("dog-and-cat-cover.jpg")
    image = image.resize((image_height, image_width), resample=Image.BICUBIC)
    image_tensor = transforms.ToTensor()(image)

    fm_g_cam = FMGradCAM(model, image_tensor, target_layer, class_count, activation_fn)
    fm_g_cam._get_saliency_maps()
    visualize_saliency_maps(fm_g_cam.saliency_maps, image)
