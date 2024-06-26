import cv2

img_gray = cv2.imread('dawg.jpg',0)

cv2.imshow('Doggo',img_gray)

cv2.waitKey(0)