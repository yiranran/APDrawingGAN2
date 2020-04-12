import cv2
import os, glob, csv, shutil
import numpy as np
import dlib
import math
from shapely.geometry import Point
from shapely.geometry import Polygon
import sys

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../checkpoints/shape_predictor_68_face_landmarks.dat')

def getfeats(featpath):
	trans_points = np.empty([68,2],dtype=np.int64) 
	with open(featpath, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		for ind,row in enumerate(reader):
			trans_points[ind,:] = row
	return trans_points

def getinternal(lm1,lm2):
	lminternal = []
	if abs(lm1[1]-lm2[1]) > abs(lm1[0]-lm2[0]):
		if lm1[1] > lm2[1]:
			tmp = lm1
			lm1 = lm2
			lm2 = tmp
		for y in range(lm1[1]+1,lm2[1]):
			x = int(round(float(y-lm1[1])/(lm2[1]-lm1[1])*(lm2[0]-lm1[0])+lm1[0]))
			lminternal.append((x,y))
	else:
		if lm1[0] > lm2[0]:
			tmp = lm1
			lm1 = lm2
			lm2 = tmp
		for x in range(lm1[0]+1,lm2[0]):
			y = int(round(float(x-lm1[0])/(lm2[0]-lm1[0])*(lm2[1]-lm1[1])+lm1[1]))
			lminternal.append((x,y))
	return lminternal

def mulcross(p,x_1,x):#p-x_1,x-x_1
	vp = [p[0]-x_1[0],p[1]-x_1[1]]
	vq = [x[0]-x_1[0],x[1]-x_1[1]]
	return vp[0]*vq[1]-vp[1]*vq[0]

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def get_68lm(imgfile,savepath):
	image = cv2.imread(imgfile)
	rgbImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	rects = detector(rgbImg, 1)
	for (i, rect) in enumerate(rects):
		landmarks = predictor(rgbImg, rect)
		landmarks = shape_to_np(landmarks)
		f = open(savepath,'w')
		for i in range(len(landmarks)):
			lm = landmarks[i]
			print(lm[0], lm[1], file=f)
		f.close()

def get_partmask(imgfile,part,lmpath,savefile):
	img = cv2.imread(imgfile)
	mask = np.zeros(img.shape, np.uint8)
	lms = getfeats(lmpath)

	if os.path.exists(savefile):
		return
	
	if part == 'nose':
		# 27,31....,35 -> up, left, right, lower5 -- eight points
		up = [int(round(1.2*lms[27][0]-0.2*lms[33][0])),int(round(1.2*lms[27][1]-0.2*lms[33][1]))]
		lower5 = [[0,0]]*5
		for i in range(31,36):
			lower5[i-31] = [int(round(1.1*lms[i][0]-0.1*lms[27][0])),int(round(1.1*lms[i][1]-0.1*lms[27][1]))]
		ratio = 2.5
		left = [int(round(ratio*lower5[0][0]-(ratio-1)*lower5[1][0])),int(round(ratio*lower5[0][1]-(ratio-1)*lower5[1][1]))]
		right = [int(round(ratio*lower5[4][0]-(ratio-1)*lower5[3][0])),int(round(ratio*lower5[4][1]-(ratio-1)*lower5[3][1]))]
		loop = [up,left,lower5[0],lower5[1],lower5[2],lower5[3],lower5[4],right]
	elif part == 'eyel':
		height = max(lms[41][1]-lms[37][1],lms[40][1]-lms[38][1])
		width = lms[39][0]-lms[36][0]
		ratio = 0.1
		gap = int(math.ceil(width*ratio))
		ratio2 = 0.6
		gaph = int(math.ceil(height*ratio2))
		ratio3 = 1.5
		gaph2 = int(math.ceil(height*ratio3))
		upper = [[lms[17][0]-2*gap,lms[17][1]],[lms[17][0]-2*gap,lms[17][1]-gaph],[lms[18][0],lms[18][1]-gaph],[lms[19][0],lms[19][1]-gaph],[lms[20][0],lms[20][1]-gaph],[lms[21][0]+gap*2,lms[21][1]-gaph]]
		lower = [[lms[39][0]+gap,lms[40][1]+gaph2],[lms[40][0],lms[40][1]+gaph2],[lms[41][0],lms[41][1]+gaph2],[lms[36][0]-2*gap,lms[41][1]+gaph2]]
		loop = upper + lower
		loop.reverse()
	elif part == 'eyer':
		height = max(lms[47][1]-lms[43][1],lms[46][1]-lms[44][1])
		width = lms[45][0]-lms[42][0]
		ratio = 0.1
		gap = int(math.ceil(width*ratio))
		ratio2 = 0.6
		gaph = int(math.ceil(height*ratio2))
		ratio3 = 1.5
		gaph2 = int(math.ceil(height*ratio3))
		upper = [[lms[22][0]-2*gap,lms[22][1]],[lms[22][0]-2*gap,lms[22][1]-gaph],[lms[23][0],lms[23][1]-gaph],[lms[24][0],lms[24][1]-gaph],[lms[25][0],lms[25][1]-gaph],[lms[26][0]+gap*2,lms[26][1]-gaph]]
		lower = [[lms[45][0]+2*gap,lms[46][1]+gaph2],[lms[46][0],lms[46][1]+gaph2],[lms[47][0],lms[47][1]+gaph2],[lms[42][0]-gap,lms[42][1]+gaph2]]
		loop = upper + lower
		loop.reverse()
	elif part == 'mouth':
		height = lms[62][1]-lms[51][1]
		width = lms[54][0]-lms[48][0]
		ratio = 1
		ratio2 = 0.2#0.1
		gaph = int(math.ceil(ratio*height))
		gapw = int(math.ceil(ratio2*width))
		left = [(lms[48][0]-gapw,lms[48][1])]
		upper = [(lms[i][0], lms[i][1]-gaph) for i in range(48,55)]
		right = [(lms[54][0]+gapw,lms[54][1])]
		lower = [(lms[i][0], lms[i][1]+gaph) for i in list(range(54,60))+[48]]
		loop = left + upper + right + lower
		loop.reverse()
		pl = Polygon(loop)

	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if part != 'mouth' and part != 'jaw':
				p = [j,i]
				flag = 1
				for k in range(len(loop)):
					if mulcross(p,loop[k],loop[(k+1)%len(loop)]) < 0:#y downside... >0 represents counter-clockwise, <0 clockwise
						flag = 0
						break
			else:
				p = Point(j,i)
				flag = pl.contains(p)
			if flag:
				mask[i,j] = [255,255,255]
	if not os.path.exists(os.path.dirname(savefile)):
		os.mkdir(os.path.dirname(savefile))
	cv2.imwrite(savefile,mask)

if __name__ == '__main__':
	imgfile = 'example/img_1701_aligned.png'
	lmfile = 'example/img_1701_aligned_68lm.txt'
	get_68lm(imgfile,lmfile)
	for part in ['eyel','eyer','nose','mouth']:
		savepath = 'example/img_1701_aligned_'+part+'mask.png'
		get_partmask(imgfile,part,lmfile,savepath)
