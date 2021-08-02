
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
    




if __name__ == "__main__":
    LoadImage(sys.argv)