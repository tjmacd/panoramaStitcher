import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def match(des1, des2):
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)
	# Apply ratio test
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append(m)
	return good

def merge(img1, img2):
	src1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
	src2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

	# Initialize SIFT detector
	sift = cv2.SIFT()
	# find keypoints and descriptors
	kp1, des1 = sift.detectAndCompute(src1, None)
	kp2, des2 = sift.detectAndCompute(src2, None)

	matches = match(des1, des2)

	dst_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
	src_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])

	H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	
	proj2 = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], 
									 img1.shape[0]))
	
	img3 = np.zeros(proj2.shape, dtype="uint8")
	img3[0:img1.shape[0], 0:img1.shape[1]] = img1
	#print proj2 > 0
	img3[proj2 > 0] = proj2[proj2 > 0]
	
	return img3



# Process command line args
parser = argparse.ArgumentParser(
	description='CSCI 4220U Project - Panorama Image Stitcher')
parser.add_argument('imgfile', nargs='+', help='Image files')
args = parser.parse_args()

imgs = []
for filename in args.imgfile:
	image = cv2.imread(filename)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	imgs.append(image)

'''
kps = []
descs = []
sift = cv2.SIFT()

for img in imgs:
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	kp, desc = sift.detectAndCompute(gray, None)
	kps.append(kp)
	descs.append(desc)
	'''

pano = imgs[0]
for img in imgs[1:]:
	pano = merge(pano, img)

plt.figure(figsize=(10,10))
for i in range(len(imgs)):
	plt.subplot(2,len(imgs),i+1)
	plt.imshow(imgs[i])

plt.subplot(212)
plt.imshow(pano)
plt.show()
