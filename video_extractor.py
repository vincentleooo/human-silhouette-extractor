from utils import model_loader

import numpy as np

import cv2
import argparse
import logging

import mmcv
from mmdet.apis import inference_detector

from tqdm import tqdm

import sys
import warnings

def argparser():
    parser = argparse.ArgumentParser(description="Video Silhouette Extractor Using Various Instance Segmentation Models")
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
    
    return opt

def main():
    warnings.filterwarnings('ignore')
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    opt = argparser()
    vid_path = opt.input
    
    video = cv2.VideoCapture(vid_path)
    
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    origin_fps = int(video.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), origin_fps, (frame_width, frame_height))
    
    model = model_loader.init(opt.model)

    pbar = tqdm(total=total_frames, unit='frames', desc='Analysing the frames')

    while (video.isOpened):
        success, img = video.read()
        
        if success:
            result = inference_detector(model, img)

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
                    
            img_show = np.zeros((frame_height, frame_width, 3))

            for i in count_list:
                img_show[segms[i]] = color_mask
            
            out.write((img_show).astype(np.uint8))
            
            pbar.update(1)
            key = cv2.waitKey(10)
            if key == 27:
                break
        else:
            break
    
    pbar.close()
    cv2.destroyAllWindows()
    video.release()
    out.release()
    
if __name__ == "__main__":
    main()