import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def match(des1, des2):
	# match features
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)
	# Apply ratio test
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append(m)
	return good

def estimateHomography(matches, kp1, kp2):
	# estimate homography
	dst_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
	src_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])

	H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	return H

'''# verify image match
def match(img1, img2):
	## detect SIFT features
	src1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
	src2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

	# Initialize SIFT detector
	sift = cv2.SIFT()
	# find keypoints and descriptors
	kp1, des1 = sift.detectAndCompute(src1, None)
	kp2, des2 = sift.detectAndCompute(src2, None)

	
	

	# get inliers
	n = len(good)
	homo_coords = np.zeros((n, 3), dtype="float32")
	for i in range(0, n):
		homo_coords[i][0] = kp2[good[i].trainIdx].pt[0]
		homo_coords[i][1] = kp2[good[i].trainIdx].pt[1]
		homo_coords[i][2] = 1
	trans_pts = np.matmul(homo_coords, H.T)
	#print trans_pts.shape
	#print trans_pts
	print good[0]
	
	# fill inliers to match info (?)
	# return true if matches and homography
	return good, H'''

def merge(img1, img2):
	
	_, H = match(img1, img2)
	
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
n = len(imgs)

# Initialize SIFT detector
sift = cv2.SIFT()

kps = []
descs = []
for img in imgs:
	## detect SIFT features
	src = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	# find keypoints and descriptors
	kp, des = sift.detectAndCompute(src, None)
	kps.append(kp)
	descs.append(des)

pw_homos = []
for i in range(n-1):
	pwmatches = match(descs[i], descs[i+1])
	H = estimateHomography(pwmatches, kps[i], kps[i+1])
	pw_homos.append(H)


# stitcher -- build_linear-simple
mid = n / 2
# set middle H to I
homos = [0 for i in range(n)]
homos[mid] = np.identity(3)
# for each adjacent image, accumulate H
for i in range(mid+1, n):
	homos[i] = np.matmul(homos[i-1], pw_homos[i-1])
for i in range(mid-1, -1, -1):
	homos[i] = np.matmul(homos[i+1], np.linalg.inv(pw_homos[i]))

plt.figure(figsize=(10,10))
for i, img in enumerate(imgs):
	proj = cv2.warpPerspective(img, homos[i], (img.shape[1]*3, img.shape[0]))
	plt.subplot(len(imgs),1,i+1)
	plt.imshow(proj)
plt.show()

#	- multiply previous by pairwise H
# f = Camera::estimate_focal(pairwise_matches)
# multiply H by [[ 1/f  0   0  ]
#				 [ 0   1/f  0  ]
#				 [ 0    0  1/f ]]
# bundle.update_proj_range()
	# homo2proj = get_homo2proj
# bundle.blend -- returns image
# crop ?

'''
	proj2 = cv2.warpPerspective(imgs[i+1], H, (imgs[i].shape[1] + imgs[i+1].shape[1], 
									 imgs[i].shape[0]))
	
	img3 = np.zeros(proj2.shape, dtype="uint8")
	img3[0:imgs[i].shape[0], 0:imgs[i].shape[1]] = imgs[i]
	#print proj2 > 0
	img3[proj2 > 0] = proj2[proj2 > 0]
	plt.figure(figsize=(10,10))
	plt.imshow(img3)
	plt.show()'''

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
plt.show()'''
