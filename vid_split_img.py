import cv2
import numpy as np
import os
from tqdm import tqdm
# import img_crop

def main(sil_vid_path):   
    # Playing video from file
    vid = cv2.VideoCapture(str(sil_vid_path))
    # vid = cv2.VideoCapture('./videos/walkman_silhh.mp4')

    try:
        if not os.path.exists('rawframes'):
            print("Creating '/rawframes' directory..")
            os.makedirs('rawframes')
        else:
            # already exists, clear
            print("'/rawframes' directory found, clearing..")
            for file in os.listdir('rawframes'):
                os.remove(os.path.join('rawframes', file))
            print("'/rawframes' cleared")
    except OSError:
        print('Error: Creating directory of rawframes')

    interval = int(vid.get(cv2.CAP_PROP_FPS)) / 2
    totalFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total Frames: {}".format(totalFrames))

    while(vid.isOpened()):
        currentFrame = 0
        success, frame = vid.read()

        if success:
            while currentFrame < totalFrames/interval:
                # Capture frame-by-frame
                success, frame1 = vid.read()

                # Saves image of the current frame in jpg file
                name = './rawframes/frame' + str(currentFrame) + '.png'
                print('Creating...' + name)
                cv2.imwrite(name, frame1)

                # To stop duplicate images
                currentFrame += 1
        vid.release()
        break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # test1.py executed as script
    # do something
    main()
    # img_crop.main('./rawframes')
    