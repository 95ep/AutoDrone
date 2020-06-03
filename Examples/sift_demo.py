import cv2 as cv
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt


class SIFTDemo:
    def __init__(self, query_paths, rejection_factor, min_match_thres):
        self.kp_q_list = []
        self.desc_q_list = []
        self.corners_list = []

        self.rejection_factor = rejection_factor
        self.min_match_count = min_match_thres
        for path in query_paths:
            query_image = cv.imread(path, cv.IMREAD_GRAYSCALE)
            h, w = query_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            sift = cv.xfeatures2d.SIFT_create()
            kp_q, desc_q = sift.detectAndCompute(query_image, None)
            self.kp_q_list.append(kp_q)
            self.desc_q_list.append(desc_q)
            self.corners_list.append(pts)

    def get_trgt_objects(self, train_path):
        train_image = cv.imread(train_path, cv.IMREAD_GRAYSCALE)

        sift = cv.xfeatures2d.SIFT_create()
        kp_t, desc_t = sift.detectAndCompute(train_image, None)
        if len(kp_t) == 0:
            global_points = np.array([], dtype=float)
            return global_points
        x = np.array([kp_t[i].pt for i in range(len(kp_t))])
        bandwidth = estimate_bandwidth(x, quantile=0.2, n_samples=500)
        if bandwidth < 0.1:
            # Not possible to form clusters
            return np.array([], dtype=float)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
        ms.fit(x)
        labels = ms.labels_
        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)

        kp_per_cluster = []
        desc_per_cluster = []
        for i in range(n_clusters):
            d, = np.where(labels == i)
            kp_per_cluster.append([kp_t[xx] for xx in d])
            desc_per_cluster.append([desc_t[xx] for xx in d])

        homographies = []
        dst_list = []

        for i in range(n_clusters):
            kp_cluster = kp_per_cluster[i]
            desc_cluster = np.array(desc_per_cluster[i], dtype=np.float32)

            # Brute force matching
            # BFMatcher with default params
            bf = cv.BFMatcher()
            good_list = []
            for descQ in self.desc_q_list:
                matches = bf.knnMatch(descQ, desc_cluster, k=2)
                # Apply ratio test
                good = []
                if len(kp_cluster) > 1:
                    for m, n in matches:
                        if m.distance < self.rejection_factor * n.distance:
                            good.append(m)
                good_list.append(good)

            # Find homography
            max_inliers = 0
            homography_tmp = None
            dst_tmp = None
            for j, good in enumerate(good_list):

                pts = self.corners_list[j]
                if len(good) > self.min_match_count:
                    src_pts = np.float32([self.kp_q_list[j][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_cluster[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                    if homography is not None:
                        matches_mask = mask.ravel().tolist()
                        dst = cv.perspectiveTransform(pts, homography)
                        non_zero_masks = np.nonzero(matches_mask)

                        no_outliers = True
                        for idx in non_zero_masks[0]:
                            if int(cv.pointPolygonTest(np.array(dst),
                                                       tuple(dst_pts[idx, 0, :].astype(np.int)), False)) \
                                    != 1:
                                no_outliers = False
                                break

                        if no_outliers and len(non_zero_masks[0]) > max_inliers:
                            max_inliers = len(non_zero_masks[0])
                            homography_tmp = homography
                            dst_tmp = dst

            if max_inliers > 0:
                dst_list.append(dst_tmp)
                homographies.append(homography_tmp)

        for dst in dst_list:
            img_boxes = cv.polylines(train_image, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
            x = int(np.sum(dst[:, 0, 0]) / 4)
            y = int(np.sum(dst[:, 0, 1]) / 4)
            img_pt = cv.circle(train_image, (x, y), 30, 255, thickness=4)

        plt.imshow(train_image, cmap='gray')
        plt.show()


if __name__ == '__main__':
    query_paths = ['D:/Exjobb2020ErikFilip/AutoDrone/imgs/ObjectDetection/screen_100.jpg',
                    'D:/Exjobb2020ErikFilip/AutoDrone/imgs/ObjectDetection/screen_101.jpg']
    sift_demo = SIFTDemo(query_paths, rejection_factor=.75, min_match_thres=10)

    # Attempt to find match object between training and reference images
    train_image = 'D:/Exjobb2020ErikFilip/AutoDrone/imgs/ObjectDetection/image102.jpg'
    sift_demo.get_trgt_objects(train_image)
