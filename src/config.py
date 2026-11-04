# Dataset configuration
NUM_CLASSES = 16
IMAGE_SIZE = 224
VALIDATION_SIZE = 0.06
RANDOM_STATE = 42


# DiT / BEiT configuration
DIT_MODEL_NAME = "microsoft/dit-base"
DIT_EPOCHS = 10
DIT_LEARNING_RATE = 3e-4
DIT_WEIGHT_DECAY = 0.05
DIT_BATCH_SIZE = 64
DIT_LABEL_SMOOTHING = 0.1
DIT_LAYER_DECAY = 0.7


# ConvNeXt configuration
CNN_MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"
CNN_EPOCHS = 8
CNN_LEARNING_RATE = 2e-4
CNN_WEIGHT_DECAY = 0.05
CNN_DROP_PATH_RATE = 0.1


# Ensemble configuration
ENSEMBLE_WEIGHT_STEP = 0.1


# Runtime configuration
DEVICE = "cuda"
