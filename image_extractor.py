from utils import model_loader

import mmcv

from mmdet.apis import inference_detector

import numpy as np
import argparse

import warnings
import time
import errno
import os

import logging
import sys

def main():
    warnings.filterwarnings('ignore')
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Image Silhouette Extractor Using Various Instance Segmentation Models")
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
        "-m", "--multiple",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Toggles detecting multiple people.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Threshold for inference detector. Default: 0.3.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='scnet-r50-fpn',
        help="Inference detector model choice. Default: 'scnet-r50-fpn'. Options: 'd-solo-light', 'scnet-r50-fpn'.",
    )

    opt = parser.parse_args()

    img_path = opt.input
    
    if not os.path.isfile(opt.input):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), opt.input
        )
    
    logging.info("Initialising the model...")
    
    model = model_loader.init(opt.model)

    start_time = time.perf_counter()
    
    logging.info("Doing inference detection...")
    result = inference_detector(model, img_path)

    logging.info("Processing the results...")
    bbox_result, segm_results = result
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)\
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_result)
    labels_impt = np.where(bboxes[:, -1] > opt.threshold)[0]

    segms = mmcv.concat_list(segm_results)

    color_mask = np.array((255, 255, 255))

    logging.info("Finding humans...")

    count = 0
    count_list = []
    for i in labels_impt:
        if labels[i] == 0:
            count_list.append(count)
            count += 1
            if not opt.multiple:
                break
        else:
            count += 1
            
    img = mmcv.imread(img_path)
    h, w, _ = img.shape
    img_show = np.zeros((h, w, 3))

    for i in count_list:
        img_show[segms[i]] = img_show[segms[i]] * 1 + color_mask * 1

    end_time = time.perf_counter()

    logging.info("Writing image...")
    mmcv.imwrite(img_show, opt.output)

    logging.info(f"{end_time - start_time:.3f} seconds taken after model initialisation.")

if __name__ == "__main__":
    main()