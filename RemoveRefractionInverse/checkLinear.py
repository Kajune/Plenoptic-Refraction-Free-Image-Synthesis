import cv2
import numpy as np
import sys

chessX = 10
chessY = 7
showImg = True

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessY*chessX,3), np.float32)
objp[:,:2] = np.mgrid[0:chessX,0:chessY].T.reshape(-1,2)

img = cv2.imread(sys.argv[1], 0)
ret, corners = cv2.findChessboardCorners(img, (chessX,chessY), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

if ret == True:
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], img.shape[::-1], None, None, 
		flags = cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST)
	
	sum_error = 0
	imgpoints2, _ = cv2.projectPoints(objp, rvecs[0], tvecs[0], mtx, dist)
	error = cv2.norm(corners, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
	sum_error += error ** 2

	print("RMSE: ", np.sqrt(sum_error))
