import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


MIN_MATCH_COUNT = 10
REJECTION_FACTOR = 0.8

queryImage = cv.imread('D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/images/mug-only-3.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
trainImage = cv.imread('D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/images/four-mugs.jpg',cv.IMREAD_GRAYSCALE) # trainImage

sift = cv.xfeatures2d.SIFT_create()

kpQ, descQ = sift.detectAndCompute(queryImage, None)
kpT, descT = sift.detectAndCompute(trainImage, None)
print(descT.shape)


done = False
bf = cv.BFMatcher()
while not done:
    matches = bf.knnMatch(descQ, descT, k=2)
    good = []
    for m,n in matches:
        if m.distance < REJECTION_FACTOR*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kpQ[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpT[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 0.2)
        if M is None:
            print("No homography")
            done = True
        else:

            matchesMask = mask.ravel().tolist()
            print(np.count_nonzero(matchesMask))

            h, w = queryImage.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts, M)

            non_zero_masks = np.nonzero(matchesMask)

            for idx in non_zero_masks[0]:
                if int(cv.pointPolygonTest(np.array(dst), tuple(dst_pts[idx,0,:].astype(np.int)), False)) != 1:
                    print("dst_pts {} not in projected polygon".format(idx))
                    done = True
                    break

            img_tmp = cv.polylines(trainImage,[np.int32(dst)],True,255,3, cv.LINE_AA)
            img_tmp = cv.circle(queryImage, (int(kpQ[0].pt[0]), int(kpQ[0].pt[1])), 10, (255, 0, 0))
            draw_params = dict(singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

            img_homography = cv.drawMatches(queryImage, kpQ, trainImage, kpT, good, None, **draw_params)
            plt.imshow(img_homography), plt.show()
            plt.clf()

            kpT_tmp = []
            descT_tmp = []
            for i, kp in enumerate(kpT):
                # Check that point is on outside
                if int(cv.pointPolygonTest(np.array(dst), kp.pt, False)) == -1:
                    kpT_tmp.append(kp)
                    descT_tmp.append(descT[i,:])
            kpT = kpT_tmp
            descT = np.array(descT_tmp)
            print(descT.shape)
    else:
        print("Too few good matches")
        done = True
