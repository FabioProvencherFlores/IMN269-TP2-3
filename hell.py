
# pip install numpy
# pip install opencv-python
import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt
import glob

cbSize = (7, 9)
imgSize = [1280, 960]
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objP = np.zeros((cbSize[0]*cbSize[1], 3), np.float32)

# real size is 20 mm, so we must multiply by 20 to account it in
objP[:, :2] = np.mgrid[0:cbSize[0], 0:cbSize[1]].T.reshape(-1, 2)*20
# print(objP)

objPts = []
imgPtsL = []
imgPtsR = []


def PrintHello():
    print("go fuck yourself")


def Calibrationnage():

    # var needed for calibration
    calibImages = glob.glob("./calibDir/*")

    for name in calibImages:
        img = cv.imread(name)
        # print(len(img), len(img[0]))
        half = len(img[0])/2
        imgL = img[:, :int(half)]
        imgR = img[:, int(half):]

        # print(len(imgL), len(imgL[0]))
        grayImgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayImgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        # cv.imshow("l", grayImgL)
        # cv.imshow("R", grayImgR)
        # cv.waitKey(3000)
        # cv.destroyAllWindows()
        foundL, cornersL = cv.findChessboardCorners(grayImgL, cbSize, None)
        foundR, cornersR = cv.findChessboardCorners(grayImgR, cbSize, None)
        # print(foundL, foundR)
        if foundR and foundL == True:
            # print("yays")
            objPts.append(objP)
            cv.cornerSubPix(grayImgL, cornersL, (11, 11), (-1, -1), criteria)
            imgPtsL.append(cornersL)

            cv.cornerSubPix(grayImgR, cornersR, (11, 11), (-1, -1), criteria)
            imgPtsR.append(cornersR)

            cv.drawChessboardCorners(imgL, cbSize, cornersL, foundL)
            cv.imshow("l", imgL)
            cv.drawChessboardCorners(imgR, cbSize, cornersR, foundR)
            cv.imshow("r", imgR)
            cv.waitKey(1000)

    cv.destroyAllWindows()

    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(
        objPts, imgPtsL, imgSize, None, None)
    heightL, widthL, channelsL = imgL.shape
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(
        cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(
        objPts, imgPtsR, imgSize, None, None)
    heightR, widthR, channelsR = imgR.shape
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(
        cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

    ########## Stereo Vision Calibration #############################################

    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same

    criteria_stereo = (cv.TERM_CRITERIA_EPS +
                       cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
        objPts, imgPtsL, imgPtsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayImgL.shape[::-1], criteria_stereo, flags)

    print("mat1", newCameraMatrixL)
    print("mat2", newCameraMatrixR)

    return "f"


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
    pointsG = sift.detect(imgG, None)
    pointsD = sift.detect(imgD, None)

    imgG = cv.drawKeypoints(imgG, pointsG, imgG)
    imgD = cv.drawKeypoints(imgD, pointsD, imgD)

    res = cv.hconcat([imgG, imgD])

    print(len(res), len(res[0]))

    cv.imshow("test", res)
    cv.waitKey(0)

    # TODO
    # extraire les points de pointsG et pointsD parce que cest des arrays weird...

    # contoursG = cv.findContours(thresh, cv.RETR_EXTERNAL)

    # findFundamentaMat prend des array de points d interets, pas tous les images
    # fondMat = cv.findFundamentalMat(a, a, cv.FM_RANSAC, ransacReprojThreshold=1)
    # print("fondamental", fondMat)


if __name__ == "__main__":
    F = Calibrationnage()
    # Process(sys.argv)
