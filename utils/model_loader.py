import mmcv
from mmcv.runner import load_checkpoint

from mmdet.models import build_detector
from torch.cuda import is_available

import errno
import os

def init():
    # Choose to use a config and initialize the detector
    config = 'configs/scnet/scnet_r50_fpn_1x_coco.py'
    
    if not os.path.isfile(config):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), config
        )

    # Setup a checkpoint file to load
    checkpoint = 'checkpoints/'\
        'scnet_r50_fpn_1x_coco-c3f09857.pth'
        
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), checkpoint
        )

    # Set the device to be used for evaluation
    device = "cuda:0" if is_available() else "cpu"

    # Load the config
    config = mmcv.Config.fromfile(config)
    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None

    # Initialize the detector
    model = build_detector(config.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = config

    # Convert the model to GPU/CPU
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()

    return model

if __name__ == "__main__":
    raise NotImplementedError