import cv2
import numpy as np

img = cv2.imread('dawg.jpg')

kernel1 = np.array([[0,0,0],
                    [0,1,0],
                    [0,0,0]])

identity = cv2.filter2D(src=img, ddepth=-1,kernel=kernel1)

# cv2.imshow('orig',img)
# cv2.imshow('identity',identity)
# cv2.imwrite('identity.jpg',identity)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

kernel2 = np.ones((5,5),np.float32)/25

img2 = cv2.filter2D(src=img, ddepth=-1,kernel=kernel2)
gauss_blur = cv2.GaussianBlur(src=img, ksize=(25,25), sigmaX=1, sigmaY=1)
median = cv2.medianBlur(src=img, ksize=5)
 
kernel3 = np.array([[0, -1,  0],
                   [-1,  5, -1],
                    [0, -1,  0]])
sharp_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel3)

cv2.imshow('orig',img)
cv2.imshow('blur',img2)
cv2.imshow('gaussian blur',gauss_blur)
cv2.imshow('median blur',median)
cv2.imshow('sharpened',sharp_img)


cv2.waitKey(0)
cv2.destroyAllWindows()
