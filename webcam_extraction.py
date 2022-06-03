
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from turtle import left

import cv2
import torch
import numpy as np
import mmcv

from mmdet.apis import inference_detector, init_detector
from mmdet.core import get_classes


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    device = torch.device(args.device)

    model = init_detector(args.config, args.checkpoint, device=device)

    camera = cv2.VideoCapture(args.camera_id)
    
    frame_width = int(camera.get(3))
    frame_height = int(camera.get(4))
    class_names = get_classes('coco')

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, img = camera.read()
        result = inference_detector(model, img)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        bbox_result, segm_results = result
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)\
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        bboxes = np.vstack(bbox_result)
        labels_impt = np.where(bboxes[:, -1] > args.score_thr)[0]

        segms = mmcv.concat_list(segm_results)

        color_mask = np.array((255, 255, 255))
        bbox_mask = np.array((0, 255, 0))
        
        count = 0
        count_list = []
        for i in labels_impt:
            if labels[i] == 0:
                count_list.append(count)
                count += 1
                if not True:
                    break
            else:
                count += 1
                
        img_show = np.zeros((frame_height, frame_width, 3))
        
        left_border_list = []
        right_border_list = []
        top_border_list = []
        bottom_border_list = []

        for i in count_list:
            img_show[segms[i]] = color_mask
            left_border = int(bboxes[i][0]) - 40
            top_border = int(bboxes[i][1]) - 40
            right_border = int(bboxes[i][2]) + 40
            bottom_border = int(bboxes[i][3]) + 40
            
            if left_border >= frame_width:
                left_border = frame_width - 1
            if right_border >= frame_width:
                right_border = frame_width - 1
            if top_border >= frame_height:
                top_border = frame_height - 1
            if bottom_border >= frame_height:
                bottom_border = frame_height - 1
                
            if left_border < 0:
                left_border = 0
            if right_border < 0:
                right_border = 0
            if top_border < 0:
                top_border = 0
            if bottom_border < 0:
                bottom_border = 0
                
            left_border_list.append(left_border)
            right_border_list.append(right_border)
            top_border_list.append(top_border)
            bottom_border_list.append(bottom_border)
            
            if len(count_list) > 1:
                img_show[top_border:bottom_border, left_border] = bbox_mask
                img_show[top_border:bottom_border, right_border] = bbox_mask
                img_show[top_border, left_border:right_border] = bbox_mask
                img_show[bottom_border, left_border:right_border] = bbox_mask
            
            # if (right_border - left_border) <= (bottom_border - top_border):
            #     padding_total = (bottom_border - top_border) - (right_border - left_border)
            #     left_border = left_border - int(padding_total / 2)
            #     right_border = right_border + int(padding_total / 2)
            #     if left_border < 0 and (right_border - left_border) < frame_width:
            #         right_border -= left_border
            #         left_border = 0
            #     elif right_border >= frame_width and (left_border - right_border) >= 0:
            #         left_border -= right_border
            #         right_border = frame_width - 1
            #     else:
            #         left_border = 0
            #         right_border = frame_width - 1
            # elif (right_border - left_border) > (bottom_border - top_border):
            #     padding_total = (right_border - left_border) - (bottom_border - top_border)
            #     top_border = top_border - int(padding_total / 2)
            #     bottom_border = bottom_border + int(padding_total / 2)
            #     if top_border < 0 and (bottom_border - top_border) < frame_width:
            #         bottom_border -= top_border
            #         top_border = 0
            #     elif bottom_border >= frame_width and (top_border - bottom_border) >= 0:
            #         top_border -= bottom_border
            #         bottom_border = frame_width - 1
            #     else:
            #         top_border = 0
            #         bottom_border = frame_width - 1
        
        try:
            top_border = min(top_border_list)
            bottom_border = max(bottom_border_list)
            left_border = min(left_border_list)
            right_border = max(right_border_list)
        except:
            top_border = 0
            bottom_border = frame_height
            left_border = 0
            right_border = frame_width
        
        cv2.imshow('frame', (img_show[top_border:bottom_border, left_border:right_border]).astype(np.uint8))


if __name__ == '__main__':
    main()