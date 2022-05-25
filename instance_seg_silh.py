import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector
from mmdet.models import build_detector
from mmdet.core import get_classes
from torch.cuda import is_available

import numpy as np
import argparse

import warnings
import time
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

def main():
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description="Silhouette Extractor Using HTC")
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="The path to the image file.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output path. Make sure the directory exists.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Threshold for inference detector. Default: 0.3.",
    )

    opt = parser.parse_args()

    img_path = opt.input
    
    if not os.path.isfile(opt.input):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), opt.input
        )
        
    if not os.path.isfile(opt.output):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), opt.output
        )
    
    model = init()

    start_time = time.perf_counter()
    result = inference_detector(model, img_path)

    bbox_result, segm_results = result
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)\
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_result)
    labels_impt = np.where(bboxes[:, -1] > opt.threshold)[0]

    classes = get_classes("coco")
    labels_impt_list = [labels[i] for i in labels_impt]
    # labels_class = [classes[i] for i in labels_impt_list]

    segms = mmcv.concat_list(segm_results)
    inds = np.where(bboxes[:, -1] > opt.threshold)[0]

    color_mask = np.array((255, 255, 255))

    count = 0
    for i in inds:
        if i == 0:
            break
        else:
            count += 1
    
    img = mmcv.imread(img_path)
    h, w, _ = img.shape
    img_show = np.zeros((h, w, 3))
    img_show[segms[count]] = img_show[segms[count]] * 1 + color_mask * 1

    end_time = time.perf_counter()

    mmcv.imwrite(img_show, opt.output)

    print(f"{end_time - start_time:.3f} seconds taken after model initialisation.")

if __name__ == "__main__":
    main()