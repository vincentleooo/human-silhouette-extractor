from utils import model_loader

import numpy as np

import cv2
import argparse
import logging
import time

import mmcv
from mmdet.apis import inference_detector

import sys
import warnings

def argparser():
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

    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
    
    prev_frame_time = 0
    new_frame_time = 0

    seg_time = 0
    # write_time = 0
    
    model = model_loader.init()
    
    while (video.isOpened):
        success, img = video.read()
        
        if success:
            logging.info("Frame {} of {}:".format(video.get(cv2.CAP_PROP_POS_FRAMES), total_frames))
        
            start_time = time.perf_counter()
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
            for i in labels_impt:
                if i == 0:
                    break
                else:
                    count += 1
                    

            h, w, _ = img.shape
            img_show = np.zeros((h, w, 3))
            img_show[segms[count]] = img_show[segms[count]] * 1 + color_mask * 1

            end_time = time.perf_counter()
            
            logging.info("Time taken to segment: {}".format(end_time - start_time))
            
            seg_time += end_time - start_time
            
            out.write((img_show).astype(np.uint8))
            
            key = cv2.waitKey(10)
            if key == 27:
                break
            
            if video.get(cv2.CAP_PROP_POS_FRAMES) == 5:
                break
        else:
            break
    
    cv2.destroyAllWindows()
    video.release()
    out.release()
    video.release()
    out.release()
    
if __name__ == "__main__":
    main()