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
for i, img in enumerate(imgs):
	## detect SIFT features
	src = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	# find keypoints and descriptors
	kp, des = sift.detectAndCompute(src, None)
	print "Detected", len(kp), "features in image", i
	kps.append(kp)
	descs.append(des)

pw_homos = []
for i in range(n-1):
	pwmatches = match(descs[i], descs[i+1])
	H = estimateHomography(pwmatches, kps[i], kps[i+1])
	pw_homos.append(H)
	print "Computed homography between images", i, "and", i+1


mid = n / 2
print "projecting homographies to image", mid
# set middle H to I
homos = [0 for i in range(n)]
homos[mid] = np.identity(3)
# for each adjacent image, accumulate H
#	- multiply previous by pairwise H
for i in range(mid+1, n):
	homos[i] = np.matmul(homos[i-1], pw_homos[i-1])
for i in range(mid-1, -1, -1):
	homos[i] = np.matmul(homos[i+1], np.linalg.inv(pw_homos[i]))


## find image boundaries
i_corners = [[0,0], [0, 1], [1, 0], [1,1]]

xMin = None
xMax = None
yMin = None
yMax = None
t_l = []
b_r = []
for i in range(len(imgs)):
	for j, corner in enumerate(i_corners):
		vec = [corner[0] * imgs[i].shape[1], 
			corner[1] * imgs[i].shape[0], 1]

		p = np.matmul(homos[i], vec )
		t_corner = [p[0]/p[2], p[1]/p[2]]
		
		if(j == 0):
			t_l.append(t_corner)
		if(j == 3):
			b_r.append(t_corner)

		xMax = max(xMax, t_corner[0])
		yMax = max(yMax, t_corner[1])
		if(xMin == None or t_corner[0] < xMin):
			xMin = t_corner[0]
		if(yMin == None or t_corner[1] < yMin):
			yMin = t_corner[1]
print "projmin:", xMin, yMin, "projmax:", xMax, yMax
out_size = (int(xMax-xMin), imgs[mid].shape[0])

# Translate homography so that projection is greater than 0
print "Translating homographies..."
T = [[1, 0, -xMin],
	 [0, 1, 0],
	 [0, 0, 1]]

for i in range(n):
	homos[i] = np.matmul(T, homos[i])

print "Projecting images"
proj = []
for i, img in enumerate(imgs):
	proj.append(cv2.warpPerspective(img, homos[i], out_size, borderMode=cv2.BORDER_CONSTANT))


print "Blending images..."

pano = np.zeros(proj[0].shape, dtype="uint8")
for i in range(mid):
	pano[proj[i] > 0] = proj[i][proj[i] > 0]
	if n-i-1 != mid:
		pano[proj[n-i-1] > 0] = proj[n-i-1][proj[n-i-1] > 0]
pano[proj[mid] > 0] = proj[mid][proj[mid] > 0]


plt.figure(figsize=(10,10))

plt.imshow(pano)
plt.show()
