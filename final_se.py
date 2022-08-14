import argparse
from xml.sax import SAXParseException
import cv2
import torch
import numpy as np
import mmcv
import os
from mmdet.apis import inference_detector, init_detector

import logging
from tqdm import tqdm
import vid_split_img
import img_crop

# from model_loader import init
# vincent added ^ to prototype

def parse_args():
    parser = argparse.ArgumentParser(description='MMdet Silhouette Extraction Cropper')
    parser.add_argument(
        "-i", "--input",
        help = "Path to the video file.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output path. Make sure the directory folder exists.",
    )
    parser.add_argument(
        "-m", "--multiple",
        type=bool,
        action=argparse.BooleanOptionalAction,
        # default= "-m",
        help="Toggles detecting multiple people.",
    )
    parser.add_argument(
        'config', default='./configs/scnet/scnet_r50_fpn_1x_coco.py' , help='test config file path'
        )
    parser.add_argument(
        'checkpoint', default = './checkpoints/scnet_r50_fpn_1x_coco-c3f09857.pth', help='checkpoint file'
        )
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option'
        )
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id'
        )
    parser.add_argument(
        '--threshold', type=float, default=0.8, help='bbox score threshold'
        )
    opt = parser.parse_args()
    return opt
# opt.output = ./video /vid_bbox.mp4
# opt.output = ./video /silvid.mp4
# opt.output = ./video /silvid_bbox.mp4
opt = parse_args()
def main():
    
    # opt = parse_args()
    video = cv2.VideoCapture(opt.input)
    device = torch.device(opt.device)
    model = init_detector(opt.config, opt.checkpoint, device=device)
    # model = init("scnet-r50-fpn")

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    origin_fps = int(video.get(cv2.CAP_PROP_FPS))

    # writes a silhouette video
    out_silvid = cv2.VideoWriter(opt.output+'/silvid.mp4', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), origin_fps, (frame_width, frame_height))

    # writes a silhouette video with a bbox
    out_silvid_b = cv2.VideoWriter(opt.output+'/silvid_bbox.mp4', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), origin_fps, (frame_width, frame_height))

    # writes a normal video with a bbox
    out_vid_b = cv2.VideoWriter(opt.output+'/vid_bbox.mp4', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), origin_fps, (frame_width, frame_height))
    
    progbar = tqdm(total=total_frames, unit='frames', desc='Analysing the frames')

    while (video.isOpened):
        success, img = video.read()

        if success:
            result = inference_detector(model, img)

            bbox_result, segm_results = result 
            labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result) 
            ]

            labels = np.concatenate(labels)
            bboxes = np.vstack(bbox_result)
            labels_impt = np.where(bboxes[:, -1] > opt.threshold)[0]

            # segms is (100, 1920, 1080)
            segms = mmcv.concat_list(segm_results)

            color_mask = np.array((255, 255, 255))
            bbox_mask = np.array((0, 255, 0))

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

            img_silvid = np.zeros((frame_height, frame_width, 3))
            img_silvid_b = np.zeros((frame_height, frame_width, 3))
            img_vid_b = img

            left_border_list = []
            right_border_list = []
            top_border_list = []
            bottom_border_list = []

            # count_list is the number of subjects
            for i in count_list:
                # segms[i] is (1920, 1080)
                img_silvid[segms[i]] = color_mask
                img_silvid_b[segms[i]] = color_mask

                # padding the bounding boxes
                # left_border = int(bboxes[i][0]) - 40
                # top_border = int(bboxes[i][1]) - 40
                # right_border = int(bboxes[i][2]) + 40
                # bottom_border = int(bboxes[i][3]) + 40
                top_border = int(bboxes[i][1])
                bottom_border = int(bboxes[i][3])

                # bbox_height = abs(top_border-bottom_border)

                # COM = int(bboxes[i][0]) + int(bboxes[i][2])/2 + (int(bboxes[i][1]) + int(bboxes[i][3]))/2
                COM = (bboxes[i][0] + bboxes[i][2])/2 
                bbox_height = abs(top_border-bottom_border)

                left_border = int(COM - 0.75*0.5*bbox_height)
                right_border = int(COM + 0.75*0.5*bbox_height)

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

            area_multi = []
            bbox_dimens = zip(left_border_list, right_border_list, top_border_list, bottom_border_list)
            for i, j, k, l in bbox_dimens:
                area_multi.append(abs(i-j)*abs(k-l))

            if len(area_multi) > 0:
                max_val = area_multi.index(max(area_multi)) 
                # gives zero because that's the first and largest person
                # if len(count_list) >= 1:
                img_silvid_b[top_border_list[max_val]:bottom_border_list[max_val], left_border_list[max_val]] = bbox_mask
                img_silvid_b[top_border_list[max_val]:bottom_border_list[max_val], right_border_list[max_val]] = bbox_mask
                img_silvid_b[top_border_list[max_val], left_border_list[max_val]:right_border_list[max_val]] = bbox_mask
                img_silvid_b[bottom_border_list[max_val], left_border_list[max_val]:right_border_list[max_val]] = bbox_mask

                img_vid_b[top_border_list[max_val]:bottom_border_list[max_val], left_border_list[max_val]] = bbox_mask
                img_vid_b[top_border_list[max_val]:bottom_border_list[max_val], right_border_list[max_val]] = bbox_mask
                img_vid_b[top_border_list[max_val], left_border_list[max_val]:right_border_list[max_val]] = bbox_mask
                img_vid_b[bottom_border_list[max_val], left_border_list[max_val]:right_border_list[max_val]] = bbox_mask   

            else:
                img_silvid_b = img_silvid_b
                img_vid_b = img_vid_b
                img_silvid = img_silvid

            out_silvid_b.write((img_silvid_b).astype(np.uint8))
            out_silvid.write((img_silvid).astype(np.uint8))
            out_vid_b.write((img_vid_b).astype(np.uint8))

            progbar.update(1)
            key = cv2.waitKey(10)

            if key == 27:
                break

        else:
            break
    # dcs
    progbar.close()
    cv2.destroyAllWindows()
    video.release()
    out_silvid_b.release()
    out_silvid.release()
    out_vid_b.release()

if __name__ == "__main__":
    main()
    vid_split_img.main(opt.output+'/silvid_bbox.mp4')
    img_crop.main('./rawframes')

