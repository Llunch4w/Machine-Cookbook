import cv2
import numpy as np

input_file = "table.jpg"
img = cv2.imread(input_file)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 初始化SIFT检测器对象并提取关键点
sift = cv2.xfeatures2d.SIFT_create()
keypoints = sift.detect(img_gray, None)
# 在输入图像上画出关键点
img_sift = np.copy(img)
cv2.drawKeypoints(img, keypoints, img_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Input image', img)
cv2.imshow('SIFT features', img_sift)
cv2.waitKey()