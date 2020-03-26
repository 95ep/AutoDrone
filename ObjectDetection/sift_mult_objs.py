import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
from sklearn.cluster import MeanShift, estimate_bandwidth

MIN_MATCH_COUNT = 10
REJECTION_FACTOR = 0.75

queryImage1 = cv.imread('D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/airsim_imgs/basic23/screen_100.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
queryImage2 = cv.imread('D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/airsim_imgs/basic23/screen_101.jpg',cv.IMREAD_GRAYSCALE)
trainImage = cv.imread('D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/airsim_imgs/basic23/image102_low_low_res.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# queryImage = cv.imread('D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/airsim_imgs/cone-only-4.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
# trainImage = cv.imread('D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/airsim_imgs/image1.jpg',cv.IMREAD_GRAYSCALE) # trainImage

queryImages = [queryImage1, queryImage2]

tot_start = time.time()
# Initiate SIFT detector
sift_start = time.time()
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kpQ1, descQ1 = sift.detectAndCompute(queryImage1, None)
kpQ2, descQ2 = sift.detectAndCompute(queryImage2, None)
kpQ_list = [kpQ1, kpQ2]
descQ_list = [descQ1, descQ2]
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

# cluster_fig = plt.figure()
# ax = cluster_fig.subplots(n_clusters//3+1,3).flatten()
# for i in range(n_clusters):
#     kpImg = cv.drawKeypoints(trainImage, kp_per_cluster[i], None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     ax[i].imshow(kpImg)
#
# plt.show()
homographies = []
dst_list = []

for i in range(n_clusters):
    kp_cluster = kp_per_cluster[i]
    desc_cluster = np.array(desc_per_cluster[i], dtype=np.float32)


    # Brute force matching
    # BFMatcher with default params
    bf = cv.BFMatcher()
    good_list = []
    n_matches = 0
    for descQ in descQ_list:
        matches = bf.knnMatch(descQ, desc_cluster, k=2)
        # Apply ratio test
        good = []
        if len(kp_cluster) > 1:
            for m,n in matches:
                if m.distance < REJECTION_FACTOR*n.distance:
                    good.append(m)
        print("n good matches {}".format(len(good)))
        if len(good) > n_matches:
            n_matches = len(good)
        good_list.append(good)


    # Find homography
    max_inliers = 0
    H_tmp = None
    dst_tmp = None
    for j, good in enumerate(good_list):
        h, w = queryImages[j].shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kpQ_list[j][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_cluster[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            if M is None:
                print("No homography")
            else:

                matchesMask = mask.ravel().tolist()

                dst = cv.perspectiveTransform(pts, M)
                non_zero_masks = np.nonzero(matchesMask)

                no_outliers = True
                for idx in non_zero_masks[0]:
                    if int(cv.pointPolygonTest(np.array(dst), tuple(dst_pts[idx,0,:].astype(np.int)), False)) != 1:
                        print("dst_pts {} not in projected polygon".format(idx))
                        no_outliers = False
                        break

                if no_outliers and len(non_zero_masks[0]) > max_inliers:
                    max_inliers = len(non_zero_masks[0])
                    H_tmp = M
                    dst_tmp = dst

                # img_tmp = cv.polylines(trainImage,[np.int32(dst)],True,255,3, cv.LINE_AA)
                # draw_params = dict(singlePointColor = None,
                #                matchesMask = matchesMask, # draw only inliers
                #                flags = 2)
                #
                # img_homography = cv.drawMatches(queryImage, kpQ, trainImage, kp_cluster, good, None, **draw_params)
                # plt.imshow(img_homography), plt.show()
                # plt.clf()
        else:
            print("Too few good matches")
            # kpImg = cv.drawKeypoints(trainImage, kp_cluster, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # plt.imshow(kpImg), plt.show()
    if max_inliers > 0:
        dst_list.append(dst_tmp)
        homographies.append(H_tmp)

tot_time = time.time() - tot_start
print("Total time {} for SIFT {}, cluster {}".format(tot_time, sift_time, cluster_time))
print("Number of objects found: {}".format(len(homographies)))

for dst in dst_list:
    img_boxes = cv.polylines(trainImage,[np.int32(dst)],True,255,3, cv.LINE_AA)
    x = int(np.sum(dst[:,0,0]) / 4)
    y = int(np.sum(dst[:,0,1]) / 4)
    img_pt = cv.circle(trainImage, (x,y), 30, 255, thickness=4)

plt.imshow(trainImage, cmap='gray'), plt.show()
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
