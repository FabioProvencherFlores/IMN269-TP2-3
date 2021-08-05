import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def drawlines(img1,img2,lines,pts1,pts2):
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

def Process():

    img = cv.imread("./images/cattest.jpg", 0)
    print(len(img), len(img[0]))
    # imgG = cv.imread('images/othertest1.jpg', 0)
    # imgD = cv.imread('images/othertest2.jpg', 0)

    moitier = len(img[0])/2
    imgG = img[:, :int(moitier)]
    imgD = img[:, int(moitier):]

    sift = cv.SIFT_create()
    keypointsG, des1 = sift.detectAndCompute(imgG,None)
    keypointsD, des2 = sift.detectAndCompute(imgD,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pointsG = []
    pointsD = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pointsD.append(keypointsD[m.trainIdx].pt)
            pointsG.append(keypointsG[m.queryIdx].pt)
    pointsG = np.int32(pointsG)
    pointsD = np.int32(pointsD)
    F, mask = cv.findFundamentalMat(pointsG,pointsD,cv.RANSAC, 3)
    print(F)
    # We select only inlier points
    pointsG = pointsG[mask.ravel()==1]
    pointsD = pointsD[mask.ravel()==1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lignes = cv.computeCorrespondEpilines(pointsD.reshape(-1,1,2), 2,F)
    lignes = lignes.reshape(-1,3)
    img5,img6 = drawlines(imgG,imgD,lignes,pointsG,pointsD)


    r,c = imgG.shape
    imgG = cv.cvtColor(imgG,cv.COLOR_GRAY2BGR)
    imgD = cv.cvtColor(imgD,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lignes,pointsG,pointsD):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        imgG = cv.line(imgG, (x0,y0), (x1,y1), color,1)
        imgG = cv.circle(imgG,tuple(pt1),5,color,-1)
        imgD = cv.circle(imgD,tuple(pt2),5,color,-1)

    

    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)


    res = cv.hconcat([img5, img6])
    cv.imshow("resultat", res)
    cv.waitKey(0)


if __name__ == "__main__":
    #F = Calibrationnage()
    #print(F)
    F= "f"
    Process()
