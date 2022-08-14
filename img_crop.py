import cv2
import os
import matplotlib.pyplot as plt

def main(rawimgs_path):
    try:
        if not os.path.exists('croppedframes'):
            print("Creating '/croppedframes' directory..")
            os.makedirs('croppedframes')
        else:
            #already exists, clear
            print("'/croppedframes directory found, clearing..")
            for file in os.listdir('croppedframes'):
                os.remove(os.path.join('croppedframes', file))
            print("'/croppedframes' cleared")
    except OSError:
        print('Error: Creating directory of croppedframes')

    for img_file in os.listdir(rawimgs_path):
        img = cv2.imread(os.path.join(rawimgs_path,img_file))

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        # dst = cv2.Canny(gray_img, 200, 200)
        # blured = cv2.blur(dst, (5,5), 0)    
        MIN_CONTOUR_AREA=200
        img_thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 3)
        Contours,imgContours = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in Contours:
            if cv2.contourArea(contour) >= MIN_CONTOUR_AREA:
                [X, Y, W, H] = cv2.boundingRect(contour)

        cropped_image = img[Y+10:Y+H-10, X+10:X+W-10]
        new_width = 64
        new_height = 64
        dim = (new_width, new_height) 
        cropped_image = cv2.resize(cropped_image, dim, interpolation= cv2.INTER_AREA)
        # print([X,Y,W,H])
        name = './croppedframes/' + str(img_file)
        plt.imshow(cropped_image)
        cv2.imwrite(name, cropped_image)

if __name__ == "__main__":
    # main('./rawframes')
    main()
    