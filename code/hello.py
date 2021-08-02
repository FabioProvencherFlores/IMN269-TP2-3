
# pip install numpy
# pip install opencv-python
import numpy as np
import cv2
import sys

def PrintHello():
    print("go fuck yourself")

def LoadImage(arg):
    img = cv2.imread(arg[1])
    print(len(img), len(img[0]))

    moitier = len(img[0])/2
    imageG = img[:,:int(moitier)]
    imageD = img[:,int(moitier)+1:]
    cv2.imshow("fenetre image", imageG)
    cv2.imshow("fenetre image2", imageD)

    cv2.waitKey(0)

    cv2.stereovision
    
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