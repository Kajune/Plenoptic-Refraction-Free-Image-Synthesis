import cv2
import numpy as np
import sys

scale_r = 1.0
scale_g = 1.0
scale_b = 1.0
offset_r = 0
offset_g = 0
offset_b = 0

img = cv2.imread(sys.argv[1])
img[:,:,0] = img[:,:,0] * scale_b + offset_b
img[:,:,1] = img[:,:,1] * scale_g + offset_g
img[:,:,2] = img[:,:,2] * scale_r + offset_r
cv2.imwrite(sys.argv[2], img)
