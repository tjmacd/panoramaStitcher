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
#	- multiply previous by pairwise H
for i in range(mid+1, n):
	homos[i] = np.matmul(homos[i-1], pw_homos[i-1])
for i in range(mid-1, -1, -1):
	homos[i] = np.matmul(homos[i+1], np.linalg.inv(pw_homos[i]))




# f = Camera::estimate_focal(pairwise_matches)
# multiply H by [[ 1/f  0   0  ]
#				 [ 0   1/f  0  ]
#				 [ 0    0  1/f ]]
# bundle.update_proj_range() -- stither_image
	# vector corner = x of Vec2D

'''
corners = []
corner_sample = 100
for i in range(corner_sample):
	for j in range(corner_sample):
		corners.append((1.0*i/corner_sample - 0.5, 1.0*j/corner_sample-0.5))
print max(corners)
print min(corners)'''

corners = [[0,0], [0, 1], [1, 0], [1,1]]

xMin = None
xMax = None
yMin = None
yMax = None
for i in range(len(imgs)):
	for j, corner in enumerate(corners):
		vec = [corner[0] * imgs[i].shape[1], 
			corner[1] * imgs[i].shape[0], 1]

		p = np.matmul(homos[i], vec )
		t_corner = [p[0]/p[2], p[1]/p[2]]
		
		xMax = max(xMax, t_corner[0])
		yMax = max(yMax, t_corner[1])
		if(xMin == None or t_corner[0] < xMin):
			xMin = t_corner[0]
		if(yMin == None or t_corner[1] < yMin):
			yMin = t_corner[1]
print "projmin:", xMin, yMin, "projmax:", xMax, yMax

T = [[1, 0, -xMin],
	 [0, 1, 0],
	 [0, 0, 1]]
print T

for i in range(n):
	homos[i] = np.matmul(T, homos[i])

proj = []
for i, img in enumerate(imgs):
	proj.append(cv2.warpPerspective(img, homos[i], (int(xMax-xMin), imgs[mid].shape[0])))

	# for each homography
		# for each corner
			# Vec homo = m.homo.trans()
			# Vec2D t_corner = homo2proj(homo) -- homogeneous to 2D

# bundle.blend -- returns image
# crop ?

pano = np.zeros(proj[0].shape, dtype="uint8")
for i in range(mid):
	pano[proj[i] > 0] = proj[i][proj[i] > 0]
	if n-i-1 != mid:
		pano[proj[n-i-1] > 0] = proj[n-i-1][proj[n-i-1] > 0]
pano[proj[mid] > 0] = proj[mid][proj[mid] > 0]


plt.figure(figsize=(10,10))
for i in range(len(imgs)):
	plt.subplot(2,len(imgs),i+1)
	plt.imshow(imgs[i])

plt.subplot(212)
plt.imshow(pano)
plt.show()
