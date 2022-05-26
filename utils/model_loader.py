import mmcv
from mmcv.runner import load_checkpoint

from mmdet.models import build_detector
from torch.cuda import is_available

import errno
import os
import logging
import sys

from utils.model_downloader import download

def init(model_choice):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if model_choice == 'd-solo-light':
        config = 'configs/solo/decoupled_solo_light_r50_fpn_3x_coco.py'
        checkpoint = 'checkpoints/'\
        'decoupled_solo_light_r50_fpn_3x_coco_20210906_142703-e70e226f.pth'
        url = "https://download.openmmlab.com/mmdetection/v2.0/solo/decoupled_solo_light_r50_fpn_3x_coco/decoupled_solo_light_r50_fpn_3x_coco_20210906_142703-e70e226f.pth"
        logging.info("Using Decoupled Solo Light model.")
    elif model_choice == 'scnet-r50-fpn':
        config = 'configs/scnet/scnet_r50_fpn_1x_coco.py'
        checkpoint = 'checkpoints/'\
        'scnet_r50_fpn_1x_coco-c3f09857.pth'
        url = "https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_r50_fpn_1x_coco/scnet_r50_fpn_1x_coco-c3f09857.pth"
        logging.info("Using SCNet R-50-FPN model.")
    else:
        raise ValueError("Model choice not found. Choose from 'd-solo-light', 'scnet-r50-fpn'.")
    
    if not os.path.isfile(config):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), config
        )
        
    if not os.path.isfile(checkpoint):
        logging.warning("Checkpoint not found. Will attempt to download from MMDetection.")
        download(url, dest_folder="checkpoints")

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