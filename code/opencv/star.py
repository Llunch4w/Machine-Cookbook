import cv2
import numpy as np
class StarFeatureDetector(object):
    def __init__(self):
        self.detector = cv2.xfeatures2d.StarDetector_create()

    def detect(self, img):
        return self.detector.detect(img)
    
input_file = "table.jpg"
input_img = cv2.imread(input_file)
img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
keypoints = StarFeatureDetector().detect(input_img)
cv2.drawKeypoints(input_img, keypoints, input_img, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Star features', input_img)

cv2.waitKey()