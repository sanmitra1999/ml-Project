import cv2
import numpy as np 

def max_width(ar):
	if len(ar)<=1:
		return 1
	c = 1
	for j in range(1,len(ar)):
		if ar[j]==ar[j-1]+1:
			c+=1
		else:
			if c>20:
				return c
			return max(c,max_width(ar[j:]))
	return c


def tup(a):
	return (a[0],a[1],a[2])


def segment_image(filename):	
	image = cv2.imread(filename)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	dic = {}
	l,w = image.shape[:2]
	for j in range(l):
		for k in range(w):
			i = tup(image[j,k,:])
			if i in dic:
				dic[i]+=1
			else:
				dic[i]=1
	lis = sorted(dic.keys(),key = lambda x:dic[x])
	liss = lis[-8:-1]
	lis = []
	for i in liss:
		if dic[i]>900:
			lis.append(i)
	liss = lis
	width = {}
	maxw = {}
	for i in liss:
		width[i]=[1600,0]
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
				te[i].append(j)
		for i in width:
			maxw[i]=max(max_width(te[i]+[700]),maxw[i])

	for i in width:
		if (width[i][1]-width[i][0]>200 and maxw[i]<20) or maxw[i]<10:
			liss.remove(i)
	
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
	chars = [np.reshape(cv2.resize(i[1],(80,80)),(80,80,1)) for i in chars]
	
	return chars