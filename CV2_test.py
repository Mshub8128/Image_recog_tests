import cv2
import numpy as np


###############################################
img = cv2.imread('Mox.jpg')

clippingLim = 200
clipMat = 32
blurType = 0
kVal = 11
maxKVal = 21
lowerThresh = 100
upperThresh = 95
maxThresh = 200

################################################

img_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# convert --> LAB color space
lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_channel, a, b = cv2.split(lab)

# do a CLAHE on L-channel
clahe = cv2.createCLAHE(clipLimit=clippingLim/100, tileGridSize=(clipMat,clipMat))
cl = clahe.apply(l_channel)

# merge CLAHEed L-channel with a / b channels
limg = cv2.merge((cl,a,b))

# Convert LAB back to to BGR colors
img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lineWindow = "Hot Diggity Dog"
cv2.namedWindow(lineWindow)

if blurType ==0:
    img_blur = cv2.medianBlur(src=img_grey, ksize=kVal)
elif blurType == 1:
    img_blur = cv2.GaussianBlur(img_grey,(kVal,kVal),0)
else:#if blurType == 2:
    img_blur = cv2.bilateralFilter(src=img_grey, d=9, sigmaColor=75, sigmaSpace=75)
 
edges = cv2.Canny(image=img_blur, threshold1=lowerThresh, threshold2=upperThresh) # Canny Edge Detection

img2= cv2.GaussianBlur(edges,(5,5),0)

imgComb = np.hstack((img_orig, img_grey, img_blur, edges ))

def thresholds(args):
    img = cv2.imread('Mox.jpg')

    clippingLim = cv2.getTrackbarPos("clipping limit",lineWindow)
    # clipMat = cv2.getTrackbarPos("tile size",lineWindow)
    blurType = cv2.getTrackbarPos("blur type",lineWindow)
    kVal = cv2.getTrackbarPos("blur",lineWindow)
    minthresh = cv2.getTrackbarPos("min",lineWindow)
    maxthresh = cv2.getTrackbarPos("max",lineWindow)
    if kVal %2 != 1:
        kVal = kVal-1

    # convert --> LAB color space
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # do a CLAHE on L-channel
    clahe = cv2.createCLAHE(clipLimit=clippingLim/100, tileGridSize=(clipMat,clipMat))
    cl = clahe.apply(l_channel)

    # merge CLAHEed L-channel with a / b channels
    limg = cv2.merge((cl,a,b))

    # Convert LAB back to to BGR colors
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    if blurType ==0:
        img_blur = cv2.medianBlur(src=img_grey, ksize=kVal)
        blurTypeName = "Median Blur"
    elif blurType == 1:
        img_blur = cv2.GaussianBlur(img_grey,(kVal,kVal),0)
        blurTypeName = "Gaussian Blur"
    else:# blurType == 2:
        img_blur = cv2.bilateralFilter(src=img_grey, d=9, sigmaColor=75, sigmaSpace=75)
        blurTypeName = "Bilateral Filter Blur"
    
    edges = cv2.Canny(image=img_blur, threshold1=minthresh, threshold2=maxthresh) # Canny Edge Detection
    img2 = cv2.GaussianBlur(edges,(5,5),0)

    imgComb = np.hstack((img_orig,img_grey, img_blur,edges))
    cv2.imshow(lineWindow, imgComb)
    cv2.imshow("test", img2)

    print(f'lower threshold: {minthresh}. Upper threshold: {maxthresh}.{blurTypeName} (@ {kVal})\nClip limit: {clippingLim/100} with matrix size {clipMat}')
 


cv2.createTrackbar("clipping limit", lineWindow, clippingLim, 500, thresholds)
# cv2.createTrackbar("tile size", lineWindow, clipMat, 33, thresholds) 
cv2.createTrackbar("blur type", lineWindow, blurType, 2, thresholds)
cv2.createTrackbar("blur", lineWindow, kVal, maxKVal, thresholds)
cv2.createTrackbar("min", lineWindow, lowerThresh, maxThresh, thresholds)
cv2.createTrackbar("max", lineWindow, upperThresh, maxThresh, thresholds)



cv2.imshow(lineWindow, imgComb)
cv2.imshow("test", img2)

cv2.waitKey(0)
 
cv2.destroyAllWindows()

########### BLOB DETECTION (ಠ_ಠ) 

# img = cv2.imread('dawg.jpg',cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('colour_test.jpg',cv2.IMREAD_GRAYSCALE)

# params = cv2.SimpleBlobDetector_Params()

# params.minThreshold = 00
# params.maxThreshold = 200

# # Filter by Area.
# params.filterByArea = True
# params.minArea = 25
 
# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1
 
#  # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.87
#  # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01
 
# detector = cv2.SimpleBlobDetector_create(params)

# keypoints = detector.detect(img)

# img_with_keypoints = cv2.drawKeypoints(img,keypoints,np.array([]),(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv2.imshow("keypoints",img_with_keypoints)
# cv2.waitKey(0)


################BLUR TYPES

# img = cv2.imread('dawg.jpg')

# kernel1 = np.array([[0,0,0],
#                     [0,1,0],
#                     [0,0,0]])

# identity = cv2.filter2D(src=img, ddepth=-1,kernel=kernel1)

# # cv2.imshow('orig',img)
# # cv2.imshow('identity',identity)
# # cv2.imwrite('identity.jpg',identity)

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# kernel2 = np.ones((5,5),np.float32)/25

# img2 = cv2.filter2D(src=img, ddepth=-1,kernel=kernel2)
# gauss_blur = cv2.GaussianBlur(src=img, ksize=(25,25), sigmaX=1, sigmaY=1)
# median = cv2.medianBlur(src=img, ksize=5)
 
# kernel3 = np.array([[0, -1,  0],
#                    [-1,  5, -1],
#                     [0, -1,  0]])
# sharp_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel3)

# bilat_filter = cv2.bilateralFilter(src=img, d=9, sigmaColor=75, sigmaSpace=75)

# cv2.imshow('orig',img)
# cv2.imshow('blur',img2)
# cv2.imshow('gaussian blur',gauss_blur)
# cv2.imshow('median blur',median)
# cv2.imshow('sharpened',sharp_img)
# cv2.imshow('bilateral',bilat_filter)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
