import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
from sklearn.cluster import MeanShift, estimate_bandwidth

MIN_MATCH_COUNT = 50
REJECTION_FACTOR = 0.7

queryImage = cv.imread('D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/images/tippex-only.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
trainImage = cv.imread('D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/images/tippex-3.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift_start = time.time()
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kpQ, descQ = sift.detectAndCompute(queryImage, None)
kpT, descT = sift.detectAndCompute(trainImage, None)
sift_time = time.time() - sift_start

cluster_start = time.time()
# Find clusters of keypoints in train image
x = np.array([kpT[i].pt for i in range(len(kpT))])

bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)

# TODO: 2 questions, bin_seeding, should it be True? Is speedup necessary??, Cluster all? Better to omitt orphans (same as outliers?)?
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
ms.fit(x)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters = len(labels_unique)
print("Number of est. clusters {}".format(n_clusters))

kp_per_cluster = []
desc_per_cluster = []
print(labels_unique)
for i in range(n_clusters):
    d, = np.where(labels == i)
    print("Number of kp in cluster {} is {}".format(i, len(d)))
    kp_per_cluster.append([kpT[xx] for xx in d])
    desc_per_cluster.append([descT[xx] for xx in d])

cluster_time = time.time() - cluster_start

cluster_fig = plt.figure()
ax = cluster_fig.subplots(n_clusters//3+1,3).flatten()
for i in range(n_clusters):
    kpImg = cv.drawKeypoints(trainImage, kp_per_cluster[i], None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ax[i].imshow(kpImg)

plt.show()
for i in range(n_clusters):
    kp_cluster = kp_per_cluster[i]
    desc_cluster = np.array(desc_per_cluster[i], dtype=np.float32)


    # Brute force matching
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descQ, desc_cluster, k=2)
    # Apply ratio test
    good = []
    if len(kp_cluster) > 1:
        for m,n in matches:
            if m.distance < REJECTION_FACTOR*n.distance:
                good.append(m)
    print("n good matches {}".format(len(good)))


    # Find homography
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kpQ[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_cluster[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        print(dst_pts[0])

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        print(M)
        if M is None:
            print("No homography")
        else:

            matchesMask = mask.ravel().tolist()

            h, w = queryImage.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts, M)

            img_tmp = cv.polylines(trainImage,[np.int32(dst)],True,255,3, cv.LINE_AA)
            draw_params = dict(singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

            img_homography = cv.drawMatches(queryImage, kpQ, trainImage, kp_cluster, good, None, **draw_params)
            plt.imshow(img_homography), plt.show()
            plt.clf()
    else:
        print("Too few good matches")
        kpImg = cv.drawKeypoints(trainImage, kp_cluster, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(kpImg), plt.show()

print("Time for SIFT {}, cluster {}".format(sift_time, cluster_time))

# Plot everything
# fig1 = plt.figure()
# ax11 = fig1.subplots()
# kpImg = cv.drawKeypoints(trainImage, kpT, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# ax11.imshow(kpImg)
#
# fig2 = plt.figure()
# ax21, ax22 = fig2.subplots(1, 2)
# ax21.imshow(img_bf)
# ax21.set_title("Brute")
# ax22.imshow(img_FLANN_matches)
# ax22.set_title("FLANN")
# fig3 = plt.figure()
# ax31 = fig3.subplots()
# ax31.imshow(img_homography)
# ax31.set_title("homography")
# plt.show()
