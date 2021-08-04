
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

            #cv.drawChessboardCorners(imgL, cbSize, cornersL, foundL)
            #cv.imshow("l", imgL)
            #cv.drawChessboardCorners(imgR, cbSize, cornersR, foundR)
            #cv.imshow("r", imgR)
            #cv.waitKey(1000)

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

    return fundamentalMatrix

def Key2Coordo(keypt):
    # for i in range(len(keypt)):
    #     if(i<10):
    #         print(np.float128(keypt[i].pt).reshape(-1, 1, 2))
    #     coordo[i] = np.float128(keypt[i].pt).reshape(-1, 1, 2)
    coordo = [k.pt for k in keypt]
    ar = np.int32(coordo)
    return ar


def Process(F):
    img = cv.imread("./images/cattest.jpg", 0)
    print(len(img), len(img[0]))
    # imgG = cv.imread('images/othertest1.jpg', 0)
    # imgD = cv.imread('images/othertest2.jpg', 0)

    moitier = len(img[0])/2
    imgG = img[:, :int(moitier)]
    imgD = img[:, int(moitier):]

    # h = 960 w = 1280
    print(len(imgG), len(imgG[0]))
    print(len(imgD), len(imgD[0]))


    # Trouver les points d'interets

    sift = cv.SIFT_create()
    keypointsG, waste1 = sift.detectAndCompute(imgG, None)
    keypointsD, waste2 = sift.detectAndCompute(imgD, None)

    # PrintPoints(keypointsD)
    tupleG = Key2Coordo(keypointsG)
    tupleD = Key2Coordo(keypointsD)


    
    # for pt in tupleG:
    #     cv.circle(imgD, pt,3,(0,0,200))
    # for pt in tupleG:
    #     cv.circle(imgG, pt,3,(0,0,200))

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(waste1,waste2,k=2)

    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(keypointsG[m.trainIdx].pt)
            pts1.append(keypointsD[m.queryIdx].pt)
    # TODO
    # extraire les points de pointsG et pointsD parce que cest des arrays weird...

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # findFundamentaMat prend des array de points d interets, pas tous les images
    fondMat, mask = cv.findFundamentalMat(tupleG, tupleD[:len(tupleG)], cv.FM_RANSAC, 3, 0.99)
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    lines = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, fondMat)
    lines = lines.reshapre(-1,3)

    r,c = imgG.shape
    imgG = cv.cvtColor(imgG,cv.COLOR_GRAY2BGR)
    imgD = cv.cvtColor(imgD,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        imgG = cv.line(imgG, (x0,y0), (x1,y1), color,1)
        imgG = cv.circle(imgG,tuple(pt1),5,color,-1)
        imgD = cv.circle(imgD,tuple(pt2),5,color,-1)
    
    #print(pointsG[:10])
    res = cv.hconcat([imgG, imgD])
    cv.imshow("resultat", res)
    cv.waitKey(0)

if __name__ == "__main__":
    #F = Calibrationnage()
    #print(F)
    F= "f"
    Process(F)
