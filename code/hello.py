
# pip install numpy
# pip install opencv-python
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

def PrintHello():
    print("go fuck yourself")


def Process(arg):
    img = cv2.imread(arg[1], 0)
    print(len(img), len(img[0]))
    # imgG = cv2.imread('images/othertest1.jpg', 0)
    # imgD = cv2.imread('images/othertest2.jpg', 0)

    moitier = len(img[0])/2
    imgG = img[:,:int(moitier)]
    imgD = img[:,int(moitier):]
    print(len(imgG), len(imgG[0]))
    print(len(imgD),len(imgD[0]))
    
    # Trouver les points d'interets

    sift = cv2.SIFT_create()
    pointsG = sift.detect(imgG,None)
    pointsD = sift.detect(imgD,None)

    imgG=cv2.drawKeypoints(imgG,pointsG,imgG)
    imgD=cv2.drawKeypoints(imgD,pointsD,imgD)
    
    res = cv2.hconcat([imgG, imgD])

    print(len(res), len(res[0]))

    cv2.imshow("test", res)
    cv2.waitKey(0)
    
    # TODO
    # extraire les points de pointsG et pointsD parce que cest des arrays weird...

    #contoursG = cv2.findContours(thresh, cv2.RETR_EXTERNAL)


    # findFundamentaMat prend des array de points d interets, pas tous les images
    #fondMat = cv2.findFundamentalMat(a, a, cv2.FM_RANSAC, ransacReprojThreshold=1)
    #print("fondamental", fondMat)




if __name__ == "__main__":
    Process(sys.argv)