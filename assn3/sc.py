import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os
from scipy.misc import toimage
from operator import itemgetter
from skimage import measure
import numpy as np



def max_width(ar):
	if len(ar)<=1:
		return 1
	c = 1
	for j in range(1,len(ar)):
		if ar[j]==ar[j-1]+1:
			c+=1
		else:
			return max(c,max_width(ar[j:]))
	return c
def tup(a):
	# print a
	return (a[0],a[1],a[2])

files = os.listdir('./train/')
# print len(files)
files =['HBNV.png'] 
for qq,im in enumerate(files):
	print qq
	# im = 'MZBI.png'
	image = cv2.imread('./train/'+im)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

	dic = {}
	l,w = image.shape[:2]
	for j in range(l):
		for k in range(w):
			11-1
			i = tup(image[j,k,:])
			if i in dic:
				dic[i]+=1
			else:
				dic[i]=1
	lis = sorted(dic.keys(),key = lambda x:dic[x])
	liss = lis[-10:-1]

	width = {}
	height = {}
	maxw = {}
	for i in liss:
		# print i,dic[i]
		width[i]=[1600,0]
		height[i]=[1600,0]
		maxw[i]=0
	l,w = image.shape[:2]
	for k in range(w):	
		maxi = 0
		te = {}
		for i in width:
			te[i]=[]
		for j in range(l):
			i = tup(image[j,k,:])
			if i in width:
				width[i]=[min(width[i][0],k),max(width[i][1],k)]
				height[i]=[min(height[i][0],j),max(height[i][1],j)]
				te[i].append(j)
		for i in width:
			maxw[i]=max(max_width(te[i]+[700]),maxw[i])

	for i in width:
		if (width[i][1]-width[i][0]>200 and maxw[i]<20) or dic[i]<900 or maxw[i]<10:
			liss.remove(i)
	lis = []


	sort = sorted(range(len(liss)),key= lambda x:width[liss[x]][0])
	for i in sort:
		lis.append(liss[i])
	for i in range(len(sort)-1):
		if width[liss[sort[i+1]]][0] < width[liss[sort[i]]][1]-5 and (width[liss[sort[i]]][1]-width[liss[sort[i]]][0]<250 or min(dic[liss[sort[i]]],dic[liss[sort[i+1]]])<2000):
			if dic[liss[sort[i]]]<dic[liss[sort[i+1]]]:
				if liss[sort[i]] in lis:
					lis.remove(liss[sort[i]])
			else:
				if liss[sort[i+1]] in lis:
					lis.remove(liss[sort[i+1]])
	# print lis

	chars = []
	for j,i in enumerate(lis):
		mask = cv2.inRange(image,np.array(i),np.array(i))
		img= cv2.bitwise_and(image, image, mask= mask)
		img = img[max(height[i][0]-4,0):min(height[i][1]+4,150),max(width[i][0]-4,0):min(width[i][1]+4,600),:]
		img = cv2.split(img)[-1]
		img = cv2.medianBlur(img,7)
		chars.append(img)

	for j,i in enumerate(chars):
		s = im[:-4]
		if len(chars)!=len(s):
			print im
			break
		s = s+'1111111'
		cv2.imshow(im,np.array(i))
		# cv2.imwrite('dataset/'+str(qq)+'char '+ str(j)+' no'+s[j]+'.png',i)
		cv2.waitKey(0)







# # th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# # th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# # cv2.imshow('binary',erosion)

# # cv2.imshow('gaussian',th3)
