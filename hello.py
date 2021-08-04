
# pip install numpy
# pip install opencv-python
import numpy as np
import cv2 as cv
import sys
import random
from matplotlib import pyplot as plt
import glob


def PrintHello():
    print("go fuck yourself")


def CalibrationnageMulti():

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objP = np.zeros((7*9, 3), np.float32)
    objP[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)*20

    # var needed for calibration
    calibImages = glob.glob("./calibDir/another/*")
    objPts = []
    imgPtsL = []
    imgPtsR = []
    camMat1 = np.zeros((3, 3), np.float32)
    camMat2 = np.zeros((3, 3), np.float32)

    # print(calibImages)

    for name in calibImages:
        img = cv.imread(name)
        # print(img)
        # print(len(img), len(img[0]))
        half = len(img[0])/2
        imgL = img[:, :int(half)]

        grayImg = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        found, cornersL = cv.findChessboardCorners(grayImg, (7, 9), None)
        if found == True:
            objPts.append(objP)

            retValL, camMatL, distL, rotationL, translationL = cv.calibrateCamera(objPts, cornersL,grayImg.shape[::-1], None, None)
            #cv.cornerSubPix(grayImg, cornersL, (11, 11), (-1, -1), criteria)

            imgPtsL.append(cornersL)


    for name in calibImages:
        img = cv.imread(name)
        half = len(img[0])/2
        imgR = img[:, int(half):]

        grayImg = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        found, cornersR = cv.findChessboardCorners(grayImg, (7, 9), None)
        if found == True:
            cv.cornerSubPix(grayImg, cornersR, (11, 11), (-1, -1), criteria)
            imgPtsR.append(cornersR)
            # cv.drawChessboardCorners(grayImg, (7, 9), cornersR, found)
            # cv.imshow("chessboard", grayImg)
            # cv.waitKey(0)
            retValR, camMatR, distR, rotationR, translationR = cv.calibrateCamera(objPts, cornersL,grayImg.shape[::-1], None, None)


    retVal, cm1, dc1, cm2, dc2, r, t, e, f = cv.stereoCalibrate(
        objPts, imgPtsL, imgPtsR, camMatL, distL, camMatR, distR, (1280, 960), None, None)

    

    print(retVal)
    print("cam mat 1: ", cm1)
    print("cam mat 2: ", cm2)

    return "f"

def CalibrageTest():
    chessboard = cv.imread("images/calibragechessboard1.jpg", 0)
    moitier = len(chessboard[0])/2
    chessboardG = chessboard[:, :int(moitier)]
    chessboardD = chessboard[:, int(moitier):]

    trouveG, coinsG = cv.findChessboardCorners(chessboardG, (7, 9), None)
    cv.drawChessboardCorners(chessboardG, (7, 9), coinsG, trouveG)
    trouveD, coinsD = cv.findChessboardCorners(chessboardD, (7, 9), None)
    cv.drawChessboardCorners(chessboardD, (7, 9), coinsD, trouveD)

    # 3D coordinates of chessboard points
    points_scene = []
    points_imageG = []
    points_imageD = []

    points_imageG.append(coinsG)
    points_imageD.append(coinsD)

    fondMat = cv.findFundamentalMat(coinsG, coinsD, cv.FM_RANSAC, ransacReprojThreshold=1)
    print(fondMat)
    res = cv.hconcat([chessboardG, chessboardD])
    cv.imshow("testing", res)
    cv.waitKey(0)

    return "f"

def PrintPoints(pts):
    for i in range(1,10):
        point = np.int32(pts[i].pt).reshape(-1, 1, 2)
        print(point)


def Key2Coordo(keypt):
    coordo = np.array([len(keypt)])
    # for i in range(len(keypt)):
    #     if(i<10):
    #         print(np.float128(keypt[i].pt).reshape(-1, 1, 2))
    #     coordo[i] = np.float128(keypt[i].pt).reshape(-1, 1, 2)
    coordo = [k.pt for k in keypt]
    
    return coordo



def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

def Process(arg):
    img = cv.imread(arg[1], 0)
    print(len(img), len(img[0]))
    # imgG = cv.imread('images/othertest1.jpg', 0)
    # imgD = cv.imread('images/othertest2.jpg', 0)

    moitier = len(img[0])/2
    imgG = img[:, :int(moitier)]
    imgD = img[:, int(moitier):]

    # h = 960 w = 1280
    print(len(imgG), len(imgG[0]))
    print(len(imgD), len(imgD[0]))

    # TODO: calibrer les shits

    # Trouver les points d'interets

    sift = cv.SIFT_create()
    keypointsG, waste1 = sift.detectAndCompute(imgG, None)
    keypointsD, waste2 = sift.detectAndCompute(imgD, None)

    # PrintPoints(keypointsD)
    tupleG = Key2Coordo(keypointsG)
    tupleD = Key2Coordo(keypointsD)

    pointsG = []
    pointsD = []
    
    for pt in tupleG:
        cv.circle(imgG,(int(pt[0]),int(pt[1])),3,random_color())
        pointsG.extend((int(pt[0]),int(pt[1])))
    for pt in tupleG:
        cv.circle(imgD,(int(pt[0]),int(pt[1])),3,random_color())
        pointsD.extend((int(pt[0]),int(pt[1])))

    print(pointsG[:10])
    res = cv.hconcat([imgG, imgD])



    cv.imshow("test", res)
    cv.waitKey(0)

    # TODO
    # extraire les points de pointsG et pointsD parce que cest des arrays weird...

    # contoursG = cv.findContours(thresh, cv.RETR_EXTERNAL)

    # findFundamentaMat prend des array de points d interets, pas tous les images
    fondMat = cv.findFundamentalMat(pointsG, pointsD[:len(pointsG)], cv.FM_RANSAC, 3)
    print("fondamental", fondMat)


if __name__ == "__main__":
    # F = CalibrageTest()
    Process(sys.argv)
