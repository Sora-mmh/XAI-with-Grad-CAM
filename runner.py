import logging
from pathlib import Path
import requests

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from PIL import Image

from data import DamagesDataset, CancerDataset
from trainer import ClassificationMobileNetV2
from eval import Evaluator
from fmgcam import FMGradCAM
from utils._base import visualize_saliency_maps
import defs

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    # sys.path.insert(0, "/fm-g-cam")
    dataset_pth = Path("/home/mmhamdi/workspace/classification/dataset/cancer_dataset/")

    classes = {"benign": 0, "malignant": 1}
    # "Missing part": 0,
    # "Broken part": 1,
    # "Scratch": 2,
    # "Dent": 3,
    # "Cracked": 4,
    # "Flaking": 5,
    # "Paint chip": 6,
    # "Corrosion": 7,

    epochs = 10
    freeze = 13
    output_pth = Path(
        "/home/mmhamdi/workspace/classification/XAI-with-fused-multi-class-Grad-CAM/outputs"
    )
    data_loaders = {}
    data_sizes = {}
    for data_type in [defs.TRAIN, defs.VAL, defs.TEST]:
        dataset = CancerDataset(dataset_pth, data_type)
        data_sizes[data_type] = dataset.__len__()
        data_loaders[data_type] = DataLoader(
            dataset, batch_size=defs.BATCH_SIZE, shuffle=(data_type == defs.TRAIN)
        )
    trainer = ClassificationMobileNetV2(
        data_loaders=data_loaders,
        data_sizes=data_sizes,
        cls=classes,
        epochs=epochs,
        freeze=freeze,
    )
    logging.info("Start building the model ...")
    trainer._build_model()
    logging.info("MobileNetV2 model built.")
    logging.info("Launch Training ...")
    trainer.train_model()
    evaluator = Evaluator(data_loaders, trainer, classes, output_pth)
    logging.info("Launch evaluation ...")
    logging.info("Plotting confusion matrix...")
    evaluator.plot_confusion_matrix()
    logging.info("Evaluation complete.")
    target_layer = trainer.model.features[18][0]
    class_count = 1
    activation_fn = F.relu
    image_width, image_height = 224, 224
    image = Image.open(
        "/home/mmhamdi/workspace/classification/dataset/cancer_dataset/test/benign/1.jpg"
    )
    image = image.resize((image_height, image_width), resample=Image.BICUBIC)
    image_tensor = transforms.ToTensor()(image)
    logging.info("Start compute saliency map with FMGradCam...")
    fm_g_cam = FMGradCAM(
        trainer.model, image_tensor, target_layer, class_count, activation_fn
    )
    fm_g_cam._get_saliency_maps()
    visualize_saliency_maps(fm_g_cam.saliency_maps, image, class_count)
    logging.info("Finished.")
