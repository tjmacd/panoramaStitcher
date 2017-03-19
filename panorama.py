import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Process command line args
parser = argparse.ArgumentParser(
	description='CSCI 4220U Project - Panorama Image Stitcher')
parser.add_argument('imgfile1', help='Image file 1')
parser.add_argument('imgfile2', help='Image file 2')
args = parser.parse_args()

img1 = cv2.imread(args.imgfile1)
img2 = cv2.imread(args.imgfile2)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT()
# find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Match using brute force matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Ratio test
good = []
for m,n in matches:
	if m.distance < 0.75*n.distance:
		good.append(m)

src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])
dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

pano = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1],
									img1.shape[0]))
pano[0:img1.shape[0], 0:img1.shape[1]] = img1

#cv2.imshow("panorama", pano)
#print 'Press any key to proceed'   
#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.figure(figsize=(10,10))
plt.subplot(221)
plt.imshow(img1)
plt.subplot(222)
plt.imshow(img2)
plt.subplot(212)
plt.imshow(pano)
plt.show()