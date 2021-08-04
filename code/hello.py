
# pip install numpy
# pip install opencv-python
import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt

def PrintHello():
    print("go fuck yourself")

def Calibrationnage():
    chessboard = cv.imread("images/calibragechessboard1.jpg",0)
    moitier = len(chessboard[0])/2
    chessboardG = chessboard[:,:int(moitier)]
    chessboardD = chessboard[:,int(moitier):]

    trouveG, coinsG = cv.findChessboardCorners(chessboardG, (7,9), None)
    cv.drawChessboardCorners(chessboardG, (7,9), coinsG, trouveG)
    trouveD, coinsD = cv.findChessboardCorners(chessboardD, (7,9), None)
    cv.drawChessboardCorners(chessboardD, (7,9), coinsD, trouveD)

    # 3D coordinates of chessboard points
    

    points_scene = []
    points_imageG = []
    points_imageD = []

    points_scene.append()
    points_imageG.append(coinsG)
    points_imageD.append(coinsD)


    res = cv.hconcat([chessboardG, chessboardD])
    cv.imshow("testing", res)
    cv.waitKey(0)

    return "f"


def Process(arg):
    img = cv.imread(arg[1], 0)
    print(len(img), len(img[0]))
    # imgG = cv.imread('images/othertest1.jpg', 0)
    # imgD = cv.imread('images/othertest2.jpg', 0)

    moitier = len(img[0])/2
    imgG = img[:,:int(moitier)]
    imgD = img[:,int(moitier):]
    print(len(imgG), len(imgG[0]))
    print(len(imgD),len(imgD[0]))
    
    # Trouver les points d'interets

    sift = cv.SIFT_create()
    pointsG = sift.detect(imgG,None)
    pointsD = sift.detect(imgD,None)

    imgG=cv.drawKeypoints(imgG,pointsG,imgG)
    imgD=cv.drawKeypoints(imgD,pointsD,imgD)
    
    res = cv.hconcat([imgG, imgD])

    print(len(res), len(res[0]))

    cv.imshow("test", res)
    cv.waitKey(0)
    
    # TODO
    # extraire les points de pointsG et pointsD parce que cest des arrays weird...

    #contoursG = cv.findContours(thresh, cv.RETR_EXTERNAL)


    # findFundamentaMat prend des array de points d interets, pas tous les images
    #fondMat = cv.findFundamentalMat(a, a, cv.FM_RANSAC, ransacReprojThreshold=1)
    #print("fondamental", fondMat)




if __name__ == "__main__":
    F = Calibrationnage()
    # Process(sys.argv)