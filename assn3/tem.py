import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os
import scipy
from operator import itemgetter
from skimage import measure
import numpy as np



files = os.listdir('./reference/')

for im in files:
	im = 'E.png'
	image = cv2.imread('./reference/'+im)
	image = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))[-1]
	mask = cv2.inRange(image,np.array([255]),np.array([255]))
	image = cv2.bitwise_and(image, image, mask= mask)
	# gray = image
	# # inv = cv2.bitwise_not(image)
	# # contours,hierarchy = cv2.findContours(inv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	# des = cv2.bitwise_not(gray)
	# contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
	# # print contour,hier
	# for cnt in contour:
	#     cv2.drawContours(des,[cnt],0,255,-1)
	# #     break
	# image = des
	cv2.imshow(im,np.array(image))
	cv2.waitKey(0)
	print set(image.ravel())
	break