import cv2
import numpy as np




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
