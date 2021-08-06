
# pip install numpy
# pip install opencv-python
import numpy as np
from numpy import float32, unravel_index
import cv2 as cv
import sys
import glob
from matplotlib import pyplot as plt
from random import randrange
from numpy import linalg
from scipy.linalg import solve

# chessboard size
cbSize = (7, 9)

# image size
imgSize = [1280, 960]

# calibration images
calibImages = glob.glob("./calibDir/*")
#imgName = "./whitebackground/testingwbg.jpg"
#imgName = "./images/testimage.jpg"
imgName = "./images/cattest.jpg"

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
    
def drawdotetline(img1, img2, pts1, pts2):
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for indx in range(len(pts1)):
    
        color = tuple(np.random.randint(0,255,3).tolist())
        p1 = pts1[indx]
        p2 = pts2[indx]
        img1 = cv.circle(img1,p1,5,color,-1)
        img2 = cv.circle(img2,p2,5,color,-1)
        img1 = cv.line(img1, p1, p2, color,1)
    return img1, img2
def drawepipipiplines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
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

def SaveImg(imgG, imgD, filename):
    res = cv.hconcat([imgG, imgD])
    cv.imwrite("./result/" + filename, res)

def FindSinglePoint(imgG, imgD, pt, pointsD, boxsize, disparityMax):
    # pour chaque poits Gauche
    # creer une sousmatrice qui inclus le voisinage
    template = ConstructWindow(imgG, pt, boxsize)
        
    
    # evaluer tous les points Droits
    maxCor = 0
    match = (0,0)
    for candidat in pointsD:
        if (2 <6 ): #(abs(pt[1] - candidat[1]) < 30) and (abs(pt[0] - candidat[0]) < disparityMax):

            sousFenetre = ConstructWindow(imgD, candidat, boxsize)
            if(len(template) > len(sousFenetre)) or len(template[0]) > len(sousFenetre[0]):
                tempTemplate = ConstructWindow(imgG, pt, min(int(len(sousFenetre)/2),int(len(sousFenetre[0])/2)))


                corMatrice = cv.matchTemplate(sousFenetre, tempTemplate, cv.TM_CCORR_NORMED)
            else:
                corMatrice = cv.matchTemplate(sousFenetre, template, cv.TM_CCORR_NORMED)
            cor = max(map(max, corMatrice))
            if cor > maxCor:
                maxCor = cor
                match = candidat
    return match, maxCor


def FindMatches(imgG, imgD, pointsG, pointsD, boxsize, disparityMax, confidenceBound):
    matchedptsD = []
    matchedptsG = []
    abberantD = []
    abberantG = []
    it = 0
    nb = 0
    for pt in pointsG:
        it += 1
        #if(it%15==0):
            #print(it)

        match, maxCor = FindSinglePoint(imgG, imgD, pt, pointsD, boxsize, disparityMax)

        #garder le maximum si un maximum est resonable
        if(abs(pt[1] - match[1]) < 30) and (abs(pt[0] - match[0]) < disparityMax) and maxCor > confidenceBound:
            matchedptsG.append(pt)
            matchedptsD.append(match)
            nb+=1
            if(nb%10==0):
                print("found ", nb, "matches so far")
        else:
            abberantG.append(pt)
            abberantD.append(match)
        
        # TODO
        # modifier le check pour inclure la disparite maximale et non la confiance
        # ca va permettre d evaluer les donnes abberante apres...


    cv.destroyAllWindows()
    np.int32(matchedptsD)
    np.int32(matchedptsG)
    np.int32(abberantG)
    np.int32(abberantD)

    return matchedptsG, matchedptsD, abberantG, abberantD




def MiseEnCorrespondance(FondMat):
    #========================================================
    #       HARDCODED PARAMS (a prendre en argument later)
    #========================================================
    confidenceBound = 0.35
    nbIteration = 10
    samplesize = 9
    windowsize = 60 #en fait juste la moitier paire du windowsize
    dispariteMax = 500


    #========================================================
    #       preparer l'image
    #========================================================

    img = cv.imread(imgName, 0)

    moitier = len(img[0])/2
    imgG = img[:, :int(moitier)]
    imgD = img[:, int(moitier):]



    print("image {", imgName, "} is loaded")

    #========================================================
    #       trouver les points d'interes
    #========================================================

    edgesG = imgG#cv.Canny(imgG, 120, 120)
    edgesD = imgD#cv.Canny(imgD, 120, 120)
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
    #PrintConcatImg(imgggg, imgddd, "test")
   



    #========================================================
    #       Mise en correspondance initiale
    #========================================================

    matchedptsG, matchedptsD, aberrantG, aberrantD = FindMatches(imgG, imgD, coordoPtsG, coordoPtsD, windowsize,dispariteMax,confidenceBound)



    resG, resD = drawdotetline(imgG, imgD, matchedptsG,matchedptsD)
    disG, disD = drawdotetline(imgG, imgD, aberrantG, aberrantD)
    #PrintConcatImg(resG,resD, "matching")
    #PrintConcatImg(disG,disD, "aberante")
    SaveImg(resG,  resD, "correlationmatchin.jpg")
    SaveImg(disG, disD, "correlationaberant.jpg")
    cv.destroyAllWindows()

        

    #========================================================
    #       depth map
    #========================================================


    # for ptG, ptD in matchedptsG, matchedptsD:
    #     line = cv.com
    pointsG = np.array(matchedptsG)
    pointsD = np.array(matchedptsD)
    n = len(pointsG)
    pt1 = np.reshape(pointsG,(1,n,2))
    pt2 = np.reshape(pointsD,(1,n,2))   

    p1, p2 = cv.correctMatches(FondMat,pt1, pt2)



    lines = cv.computeCorrespondEpilines(pointsG.reshape(-1,1,2), 2,FondMat)
    lines = lines.reshape(-1,3)
    resG,resD = drawepipipiplines(imgG,imgD,lines,pointsG,matchedptsD)
    SaveImg(resG,resD,"droiteepipolaire.jpg")

    return imgG, imgD, matchedptsG, matchedptsD

def Disparite(t1, t2):

    img = cv.imread(imgName, 0)

    moitier = len(img[0])/2
    imgG = img[:, :int(moitier)]
    imgD = img[:, int(moitier):]

    	# Setting parameters for StereoSGBM algorithm
    minDisparity = 100
    maxDisparity = 300
    numDisparities = maxDisparity-minDisparity
    blockSize =3
    disp12MaxDiff = 1
    uniquenessRatio = 5
    speckleWindowSize =10000
    speckleRange = 50
	 
	# Creating an object of StereoSGBM algorithm
    stereomatcher = cv.StereoSGBM_create(minDisparity = minDisparity,
            numDisparities = numDisparities,
            blockSize = blockSize,
            disp12MaxDiff = disp12MaxDiff,
            uniquenessRatio = uniquenessRatio,
            speckleWindowSize = speckleWindowSize,
            speckleRange = speckleRange,
            P1=8 * 1 * blockSize * blockSize,
            P2=32 * 1 * blockSize * blockSize,
        )

    disparityMap = stereomatcher.compute(imgG, imgD).astype(np.float32)
    disparityMap = cv.normalize(disparityMap, disparityMap,alpha = 255,beta = 0,norm_type= cv.NORM_MINMAX)
    cv.imshow("disp", disparityMap)
    cv.waitKey(0)
    cv.imwrite("result/disparitymap.jpg", disparityMap)





if __name__ == "__main__":
    #F = Calibrationnage()
    i1, i2 = 1, 2
    #i1, i2, p1, p2 = MiseEnCorrespondance(F)   

    Disparite(i1, i2)

