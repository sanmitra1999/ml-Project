import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os
import scipy
from operator import itemgetter
from skimage import measure
import numpy as np


# def ang(s):
# 	flag=s/abs(s)
# 	flag*=-1
# 	s = abs(s)
# 	if s<0.03:
# 		return 0
# 	if s<0.05:
# 		return 10*flag
# 	if s<0.25 :
# 		return 20*flag
# 	return 30*flag



def bwim(gray):
	gray = cv2.split(cv2.cvtColor(gray, cv2.COLOR_RGB2HSV))[-1]
	mask = cv2.inRange(gray,np.array([255]),np.array([255]))
	gray = cv2.bitwise_and(gray, gray, mask= mask)
	l,r = gray.shape
	contours,hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for i in contours:
		x,y,q,h = cv2.boundingRect(i)
		img = gray[max(y-5,0):min(y+h+5,l),max(x-5,0):min(x+q+5,w)]
		new_height = int((1.0 * img.shape[0])/img.shape[1] * 150.0)
		return cv2.resize(img, (150, new_height))
	

def ang(c):
	if c<=-45 and c>-55:
		return 10
	if c<=-55 and c>-65:
		return 20
	if c<=-65 and c>-75:
		return 30
	if c<=-75 and c>-85:
		return 40
	if c<= -5 and c>=-15:
		return -10
	if c<= -15 and c>=-25:
		return -20
	if c<=-25 and c>-35:
		return -30
	if c<=-35 and c>-45:
		return -40
	
	return 0


def angl(img):
	contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)
	for i in contours:
		a,b,c = cv2.minAreaRect(i)
		z = ang(c)
		# print b,c,z
		return b,z

def deskew(img):
	m = cv2.moments(img)
	l,r = img.shape
	# l=140-l
	# r=140-r
	# a=l/2
	# b=l-a
	# c=r/2
	# d=r-c
	# img =cv2.copyMakeBorder(img.copy(),20,20,20,20,cv2.BORDER_CONSTANT,value=0)
	l,r = img.shape
	if abs(m['mu02']) < 1e-2:
		# no deskewing needed. 
		return img.copy()
	# Calculate skew based on central momemts. 
	skew = m['mu11']/m['mu02']
	cX = int(m["m10"] / m["m00"])
	cY = int(m["m01"] / m["m00"])
	# print skew,ang(skew)
	# Calculate affine transform to correct skewness. 
	# M = np.float32([[1, skew, -0.5*l*skew], [0, 1, 0]])
	# # Apply affine transform
	# img = cv2.warpAffine(img, M, (l, w), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	# rot_mat = cv2.getRotationMatrix2D((cX,cY), ang(skew), 1.0)
	c,d = angl(img)
	# print skew,d
	rot_mat = cv2.getRotationMatrix2D(c,d, 1.0)
	img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
	return img

def rot(im,theta):
	l,r = im.shape
	l=140-l
	r=140-r
	a=l/2
	b=l-a
	c=r/2
	d=r-c
	im =cv2.copyMakeBorder(im,20,20,20,20,cv2.BORDER_CONSTANT,value=0)
	c,d = angl(im)
	rot_mat = cv2.getRotationMatrix2D(c,theta, 1.0)
	gray = cv2.warpAffine(im, rot_mat, im.shape[1::-1], flags=cv2.INTER_LINEAR)
	l,r = gray.shape
	contours,hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for i in contours:
		x,y,q,h = cv2.boundingRect(i)
		img= gray[max(y-5,0):min(y+h+5,l),max(x-5,0):min(x+q+5,w)]
		new_height = int((1.0 * img.shape[0])/img.shape[1] * 150.0)
		return cv2.resize(img, (150, new_height))


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

def cord(ar):
	ans = []
	for i in range(len(ar)):
		if ar[i]!=0:
			if len(ans)%2==0:
				ans.append(i)
			elif i+1==len(ar) or ar[i+1]==0:
				ans.append(i)
	return ans


files = os.listdir('./train/')
files = ['AVXS.png','IJSJ.png','EJT.png','OGX.png','KDSK.png']
# print len(files)
for qq,im in enumerate(files[:1]):
	print qq
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
	# lis = []


	# sort = sorted(range(len(liss)),key= lambda x:width[liss[x]][0])
	# for i in sort:
	# 	lis.append(liss[i])
	# 	# print liss[i],dic[liss[i]]
	# # for i in range(len(sort)-1):
	# # 	if width[liss[sort[i+1]]][0] < width[liss[sort[i]]][1]-5 and (width[liss[sort[i]]][1]-width[liss[sort[i]]][0]<250 or min(dic[liss[sort[i]]],dic[liss[sort[i+1]]])<2000):
	# # 		print width[liss[sort[i+1]]][0], width[liss[sort[i]]][1]
	# # 		if dic[liss[sort[i]]]<dic[liss[sort[i+1]]]:
	# # 			if liss[sort[i]] in lis:
	# # 				lis.remove(liss[sort[i]])
	# # 		else:
	# # 			if liss[sort[i+1]] in lis:
	# # 				lis.remove(liss[sort[i+1]])
	chars = []
	for j,i in enumerate(liss):
		if j==0:
			mask = cv2.inRange(image,np.array(i),np.array(i))
		else:
			mask = cv2.bitwise_or(mask,cv2.inRange(image,np.array(i),np.array(i)))
	img= cv2.bitwise_and(image, image, mask= mask)
	img = cv2.split(img)[-1]
	img = cv2.medianBlur(img,7)
	mask = cv2.inRange(img,np.array([255]),np.array([255]))
	img = cv2.bitwise_and(img, img, mask= mask)
	
	contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for i in contours:
		x,y,q,h = cv2.boundingRect(i)
		temp = img[max(y-5,0):min(y+h+5,l),max(x-5,0):min(x+q+5,w)]
		chars.append([x,temp])
	chars = sorted(chars,key = lambda x:x[0])
	chars = [i[1] for i in chars]
	
	print im
	
	for j,i in enumerate(chars):
		s = im[:-4]
		if len(chars)!=len(s):
			print im
			break

		s = s+'1111111'
		# cv2.imwrite('dataset/'+im+' '+ str(j)+' no'+s[j]+'.png',i)
		# print cv2.matchShapes(i,chars[1],cv2.CONTOURS_MATCH_I3,1)
		file = os.listdir('./reference/')
		for im0 in file:
			image = cv2.imread('./reference/'+im0)
			# image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			kim = bwim(image)
			cv2.imshow(im0,np.array(kim))
			cv2.waitKey(0)
			mini = 1
			for j in [-30,-20,-10,0,10,20,30]:
				mini = min(cv2.matchShapes(rot(i,j),kim,cv2.CONTOURS_MATCH_I2,1),mini)
			# print im0,j,mini
			# break
		# cv2.imshow(im,np.array(i))
		# cv2.waitKey(0)
		break
		# cv2.imshow(im,np.array(deskew(i)))
		# cv2.waitKey(0)

