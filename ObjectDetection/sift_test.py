import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time


MIN_MATCH_COUNT = 10
REJECTION_FACTOR = 0.70

queryImage = cv.imread('D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/airsim_imgs/cone-only-4.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
trainImage = cv.imread('D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/airsim_imgs/image3.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# queryImage = cv.imread('D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/images/tippex-only.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
# trainImage = cv.imread('D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/images/tippex-2.jpg',cv.IMREAD_GRAYSCALE) # trainImage

# Initiate SIFT detector
sift_start = time.time()
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kpQ, descQ = sift.detectAndCompute(queryImage, None)
kpT, descT = sift.detectAndCompute(trainImage, None)
sift_time = time.time() - sift_start


# Brute force matching
# BFMatcher with default params
bf_start = time.time()
bf = cv.BFMatcher()
matches = bf.knnMatch(descQ, descT,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < REJECTION_FACTOR*n.distance:
        good.append(m)
bf_time = time.time() - bf_start
# cv.drawMatchesKnn expects list of lists as matches.
img_bf = cv.drawMatches(queryImage, kpQ, trainImage, kpT, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


flann_start = time.time()
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(descQ, descT, k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < REJECTION_FACTOR*n.distance:
        matchesMask[i]=[1,0]
flann_time = time.time() - flann_start

draw_params = dict(matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_FLANN_matches = cv.drawMatchesKnn(queryImage, kpQ, trainImage, kpT, matches, None, **draw_params)

homo_start = time.time()
# Find homography
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kpQ[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpT[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h,w = queryImage.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)

    img_tmp = cv.polylines(trainImage,[np.int32(dst)],True,255,3, cv.LINE_AA)
    draw_params = dict(singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img_homography = cv.drawMatches(queryImage, kpQ, trainImage, kpT, good, None, **draw_params)

else:
    img_homography = np.zeros((50,50))
    print("Too few matches to find homography. n matches = {}".format(len(good)))
homo_time = time.time() - homo_start
print("Time for SIFT {}, BF {}, FLANN {} and homography {}".format(sift_time, bf_time, flann_time, homo_time))

# Plot everything
fig1 = plt.figure()
ax11 = fig1.subplots()
kpImg = cv.drawKeypoints(trainImage, kpT, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
ax11.imshow(kpImg)

fig2 = plt.figure()
ax21, ax22 = fig2.subplots(1, 2)
ax21.imshow(img_bf)
ax21.set_title("Brute")
ax22.imshow(img_FLANN_matches)
ax22.set_title("FLANN")
fig3 = plt.figure()
ax31 = fig3.subplots()
ax31.imshow(img_homography)
ax31.set_title("homography")
plt.show()
