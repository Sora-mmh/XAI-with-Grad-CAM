# Dataset types
TRAIN = "train"
VAL = "val"
TEST = "test"

# Data loading configuration
NUM_WORKERS = 4
BATCH_SIZE = 32

# Classes
cls = {
    "Missing part": 0,
    "Broken part": 1,
    "Scratch": 2,
    "Cracked": 3,
    "Dent": 4,
    "Flaking": 5,
    "Paint chip": 6,
    "Corrosion": 7,
}

# Image target size
TARGET_SIZE = 227

# Number of channels
NUM_CHANNELS = 3

# Normalization config
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

WEIGHTS_DIR_PTH = (
    "/home/mmhamdi/workspace/classification/XAI-with-fused-multi-class-Grad-CAM/weights"
)
