# Human Silhouette Extractor Using MMDetection [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a tool to extract silhouettes from images or videos using [MMDetection](https://github.com/open-mmlab/mmdetection) through instance segmentation as opposed to the more commonly used semantic segmentation. It uses the first model shown in the [SCNet](https://github.com/open-mmlab/mmdetection/tree/master/configs/scnet) page by default. Currently providing out-of-the-box support for the [Decoupled Light SOLO](https://github.com/open-mmlab/mmdetection/tree/master/configs/solo) model as well for faster performance with the caveat that the silhouettes are far less refined in the edges.

An example for images is shown below.

Original Image            |  Silhouette
:-------------------------:|:-------------------------:
![walking](https://user-images.githubusercontent.com/55942045/170315573-331fa0bb-46b3-4d1e-9381-f0b80f5de9ef.jpg) |  ![walking-output](https://user-images.githubusercontent.com/55942045/170315590-07a25433-1691-49e5-9881-5daf0e6e5fb8.jpg)

Image credits to [Verywell Fit](https://www.verywellfit.com/how-far-can-a-healthy-person-walk-3975556).

## Setup

1. Install MMDetection as outlined in [this tutorial](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md).
2. Install `tqdm` in the virtual environment used to install MMDetection.
3. Clone this repository.

## Usage

The use of the tools will mainly be through the CLI. If this is the first time running, the scripts will download the model checkpoint to the `checkpoints` folder.

### Video Extractor

This will use `video_extractor.py`.

    $ python video_extractor.py -h

    usage: video_extractor.py [-h] [-i INPUT] [-o OUTPUT] [-m | --multiple | --no-multiple] [--threshold THRESHOLD] [--model MODEL]

    Video Silhouette Extractor Using Various Instance Segmentation Models

    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
                            The path to the image file.
      -o OUTPUT, --output OUTPUT
                            Output path. Make sure the directory exists.
      -m, --multiple, --no-multiple
                            Toggles detecting multiple people.
      --threshold THRESHOLD
                            Threshold for inference detector. Default: 0.3.
      --model MODEL         Inference detector model choice. Default: 'scnet-r50-fpn'. Options: 'd-solo-light', 'scnet-r50-fpn'.

### Image Extractor

This will use `image_extractor.py`.

    $ python image_extractor.py
    
    usage: image_extractor.py [-h] [-i INPUT] [-o OUTPUT] [-m | --multiple | --no-multiple] [--threshold THRESHOLD] [--model MODEL]

    Image Silhouette Extractor Using Various Instance Segmentation Models

    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
                            The path to the image file.
      -o OUTPUT, --output OUTPUT
                            Output path. Make sure the directory exists.
      -m, --multiple, --no-multiple
                            Toggles detecting multiple people.
      --threshold THRESHOLD
                            Threshold for inference detector. Default: 0.3.
      --model MODEL         Inference detector model choice. Default: 'scnet-r50-fpn'. Options: 'd-solo-light', 'scnet-r50-fpn'.
                            
## Acknowledgement

We would like to acknowledge the MMDetection, SCNet, and SOLO team for their great work, and also the insights gained from a long-closed [issue](https://github.com/open-mmlab/mmdetection/issues/248) in MMDetection. Everything in the `configs` folder is attributed to [MMDetection](https://github.com/open-mmlab/mmdetection).
