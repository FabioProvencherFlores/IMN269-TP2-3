
# pip install numpy
# pip install opencv-python
import numpy as np
from numpy import unravel_index
import cv2 as cv
import sys
import glob
from matplotlib import pyplot as plt
from random import randrange
from scipy import signal

# chessboard size
cbSize = (7, 9)

# image size
imgSize = [1280, 960]

# calibration images
calibImages = glob.glob("./calibDir/*")

# criteria for opencv calibration
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objP = np.zeros((cbSize[0]*cbSize[1], 3), np.float32)

# object points in 3D coordinates
# real size is 20 mm, so we must multiply by 20 to account it in
objP[:, :2] = np.mgrid[0:cbSize[0], 0:cbSize[1]].T.reshape(-1, 2)*20

# array for object points, left and right image points
objPts = []
imgPtsL = []
imgPtsR = []


def Calibrationnage():

    # iterates through calibration picture to augment precision
    for name in calibImages:

        # splits whole image into left and right
        img = cv.imread(name)
        half = len(img[0])/2
        imgL = img[:, :int(half)]
        imgR = img[:, int(half):]

        # conversion to gray level
        grayImgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayImgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        # for calibration
        foundL, cornersL = cv.findChessboardCorners(grayImgL, cbSize, None)
        foundR, cornersR = cv.findChessboardCorners(grayImgR, cbSize, None)

        if foundR and foundL == True:

            # if the chessboard pattern was found, add data to object points, left and right image points
            objPts.append(objP)

            # this step is to further augment precision on the computing of fundamental matrix
            cv.cornerSubPix(grayImgL, cornersL, (11, 11), (-1, -1), criteria)
            imgPtsL.append(cornersL)

            cv.cornerSubPix(grayImgR, cornersR, (11, 11), (-1, -1), criteria)
            imgPtsR.append(cornersR)

    # The actual calibration of left camera
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(
        objPts, imgPtsL, imgSize, None, None)
    heightL, widthL, channelsL = imgL.shape
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(
        cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

    # the actual calibration of right camera
    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(
        objPts, imgPtsR, imgSize, None, None)
    heightR, widthR, channelsR = imgR.shape
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(
        cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

    # Stereo vision calibration
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC

    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same
    criteria_stereo = (cv.TERM_CRITERIA_EPS +
                       cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
        objPts, imgPtsL, imgPtsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayImgL.shape[::-1], criteria_stereo, flags)
    print("retval:\n", retStereo)
    print("\nRight camera intrinsic:\n", newCameraMatrixL)
    print("\nLeft camera intrinsic:\n", newCameraMatrixR)
    print("\nFundamental:\n", fundamentalMatrix)
    print("\nEssential:\n", essentialMatrix)
    print("\nTranslation:\n", trans)
    print("\nRotation:\n", rot)
    print("\nDistorsion left:\n", distL)
    print("\nDistorsion right:\n", distR)

    return fundamentalMatrix


def Key2Coordo(keypt):
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
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(waste1, waste2, k=2)

    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(keypointsG[m.trainIdx].pt)
            pts1.append(keypointsD[m.queryIdx].pt)
    # TODO
    # extraire les points de pointsG et pointsD parce que cest des arrays weird...

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # findFundamentaMat prend des array de points d interets, pas tous les images
    fondMat, mask = cv.findFundamentalMat(
        tupleG, tupleD[:len(tupleG)], cv.FM_RANSAC, 3, 0.99)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    lines = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fondMat)
    lines = lines.reshapre(-1, 3)

    r, c = imgG.shape
    imgG = cv.cvtColor(imgG, cv.COLOR_GRAY2BGR)
    imgD = cv.cvtColor(imgD, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        imgG = cv.line(imgG, (x0, y0), (x1, y1), color, 1)
        imgG = cv.circle(imgG, tuple(pt1), 5, color, -1)
        imgD = cv.circle(imgD, tuple(pt2), 5, color, -1)

    # print(pointsG[:10])
    res = cv.hconcat([imgG, imgD])
    cv.imshow("resultat", res)
    cv.waitKey(0)

    # lignes = cv.computeCorrespondEpilines(pointsD.reshape(-1,1,2), 2,F)
    # lignes = lignes.reshape(-1,3)


# Copier de la doc
# Voir https://docs.opencv.org/4.5.2/da/de9/tutorial_py_epipolar_geometry.html
def drawlines(img1,img2, pts1,pts2):

    for indx in range(len(pts1)):
        
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = pts1[indx]
        x1,y1 = pts2[indx]
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
    return img1,img2

def drawdot(img1,img2,pts1,pts2):
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for pt1,pt2 in zip(pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def ConstructWindow(img, coordo, sizeT):
    #reduit la taille de la fenetre sur les bords
    Xlow = max(coordo[0]-sizeT, 0)
    Xhigh = min(coordo[0]+ sizeT, len(img[0]))
    Ylow = max(coordo[1]-sizeT, 0)
    Yhigh = min(coordo[1]+ sizeT, len(img[0]))


    template = img[Ylow:Yhigh, Xlow:Xhigh]
    return template

def PrintConcatImg(imgG, imgD, title):
    res = cv.hconcat([imgG, imgD])
    cv.imshow(title, res)
    cv.waitKey(0)
    cv.destroyAllWindows()

def RunRansac():
    #========================================================
    #       HARDCODED PARAMS (a prendre en argument later)
    #========================================================
    confidenceBound = 0.55
    nbIteration = 10
    samplesize = 2
    windowsize = 60 #en fait juste la moitier paire du windowsize
    dispariteMax = 500
    searchwindow = 200


    #========================================================
    #       preparer l'image
    #========================================================
    imgName = "./whitebackground/testingwbg.jpg"
    img = cv.imread(imgName, 0)

    moitier = len(img[0])/2
    imgG = img[:, :int(moitier)]
    imgD = img[:, int(moitier):]



    print("image {", imgName, "} is loaded")

    #========================================================
    #       trouver les points d'interes
    #========================================================

    edgesG = cv.Canny(imgG, 120, 120)
    edgesD = cv.Canny(imgD, 120, 120)
    print(len(edgesD),len(edgesD[0]))



    sift = cv.SIFT_create()
    keypointsG, waste1 = sift.detectAndCompute(edgesG, None)
    keypointsD, waste2 = sift.detectAndCompute(edgesD, None)

    coordoPtsG = Key2Coordo(keypointsG)
    coordoPtsD = Key2Coordo(keypointsD)
    nbPtsG = len(coordoPtsG)
    nbPtsD = len(coordoPtsD)
    print("found (", nbPtsG,",",nbPtsD, ") points d'interet")
    print("for example: ", coordoPtsD[nbPtsD-3])

    imgggg, imgddd = drawdot(edgesG, edgesD, coordoPtsG, coordoPtsD)
    PrintConcatImg(imgggg, imgddd, "test")
   



    #========================================================
    #       Mise en correspondance initiale
    #========================================================


    # paddedimgG = np.pad(edgesG, ((windowsize, windowsize), (windowsize,windowsize)), 'constant', constant_values=((0,0),(0,0)))

    matchedptsD = []
    matchedptsG = []
    abberantD = []
    abberantG = []
    it = 0
    nb = 0
    for pt in coordoPtsG:
        it += 1
        if(it%15==0):
            print(it)

        # pour chaque poits Gauche
        # creer une sousmatrice qui inclus le voisinage
        template = ConstructWindow(imgG, pt, windowsize)
            
        
        # evaluer tous les points Droits
        maxCor = 0
        match = (0,0)
        for candidat in coordoPtsD:
            if(abs(pt[1] - candidat[1]) < 30) and (abs(pt[0] - candidat[0]) < dispariteMax):

                sousFenetre = ConstructWindow(imgD, candidat, windowsize)
                if(len(template) > len(sousFenetre)) or len(template[0]) > len(sousFenetre[0]):
                    tempTemplate = ConstructWindow(imgG, pt, min(int(len(sousFenetre)/2),int(len(sousFenetre[0])/2)))


                    corMatrice = cv.matchTemplate(sousFenetre, tempTemplate, cv.TM_CCORR_NORMED)
                else:
                    corMatrice = cv.matchTemplate(sousFenetre, template, cv.TM_CCORR_NORMED)
                cor = max(map(max, corMatrice))
                if cor > maxCor:
                    maxCor = cor
                    match = candidat

        #garder le maximum si un maximum est resonable
        if(maxCor > confidenceBound):
            matchedptsG.append(pt)
            matchedptsD.append(match)
            nb+=1
            if(nb%10==0):
                print("found ", nb, "matches so far")


    cv.destroyAllWindows()
    np.int32(matchedptsD)
    np.int32(matchedptsG)



    cormatG, cormatD = drawdot(imgG, imgD, matchedptsG,matchedptsD)
    resG, resD = drawlines(cormatG, cormatD, matchedptsG,matchedptsD)
    
    PrintConcatImg(resG,resD, "matching")
    # cormatG, cormatD = drawdot(edgesG, imgD, abberantG,abberantD)
    # PrintConcatImg(cormatG,cormatD, "matching")
    # cormatG, cormatD = drawdot(imgG, imgD, matchedptsG,matchedptsD)
    # PrintConcatImg(cormatG,cormatD, "matching")
    cv.destroyAllWindows()


        


    #========================================================
    #       Mise en correspondance avec un sous-ensemble aleatoire
    #========================================================

    # randomSampledPts = []
    # for i in range(samplesize):
    #     candidat = coordoPtsG[randrange(nbPtsG)]
    #     randomSampledPts.append(candidat)
    #     template = ConstructTemplate(paddedimgG, candidat, windowsize) 
        




    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(waste1, waste2, k=2)

    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < 0.8*n.distance:
    #         pts2.append(keypointsG[m.trainIdx].pt)
    #         pts1.append(keypointsD[m.queryIdx].pt)
    # TODO
    # extraire les points de pointsG et pointsD parce que cest des arrays weird...

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return "f"


if __name__ == "__main__":
    #F = Calibrationnage()
    # print(F)
    F = "f"
    #Process(F)
    RunRansac()   

