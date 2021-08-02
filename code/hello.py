
# pip install numpy
# pip install opencv-python
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

def PrintHello():
    print("go fuck yourself")

def LoadImage(arg):
    img = cv2.imread(arg[1], 0)
    print(len(img), len(img[0]))
    # imgG = cv2.imread('images/othertest1.jpg', 0)
    # imgD = cv2.imread('images/othertest2.jpg', 0)

    moitier = len(img[0])/2
    imgG = img[:,:int(moitier)]
    imgD = img[:,int(moitier):]
    print(len(imgG), len(imgG[0]))
    print(len(imgD),len(imgD[0]))
    stereo = cv2.StereoBM_create(numDisparities=80, blockSize=11)
    disparity = stereo.compute(imgG,imgD)
    plt.figure(figsize = (20,10))
    plt.imshow(disparity,'Purples')
    plt.xticks([])
    plt.yticks([])

    plt.show()

    #cv2.waitKey(0)
    
    # for filename in images:
    #     image = cv2.imread(filename)
    #     grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #     ret, corners = cv2.findChessboardCorners(
    #                     grayColor, CHECKERBOARD,
    #                     cv2.CALIB_CB_ADAPTIVE_THRESH
    #                     + cv2.CALIB_CB_FAST_CHECK +
    #                     cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    #     if ret == True:
    #         threedpoints.append(objectp3d)
    
    #         # Refining pixel coordinates
    #         # for given 2d points.
    #         corners2 = cv2.cornerSubPix(
    #             grayColor, corners, (11, 11), (-1, -1), criteria)
    
    #         twodpoints.append(corners2)
    
    #         # Draw and display the corners
    #         image = cv2.drawChessboardCorners(image,
    #                                           CHECKERBOARD,
    #                                           corners2, ret)
    
    #     cv2.imshow('img', image)
    #     cv2.waitKey(0)



if __name__ == "__main__":
    LoadImage(sys.argv)