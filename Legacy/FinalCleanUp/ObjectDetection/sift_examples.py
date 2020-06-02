import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/airsim_imgs/basic23/screen_100.jpg', 0) # query img
img2 = cv.imread('D:/Exjobb2020ErikFilip/AutoDrone/ObjectDetection/airsim_imgs/basic23/image102.jpg', 0) # train img

# init SIFT detector
sift = cv.xfeatures2d.SIFT_create()

# find keypoints and descriptors with SIFT
kp1, desc1 = sift.detectAndCompute(img1, None)
kp2, desc2 = sift.detectAndCompute(img2, None)

# Create BFMatcher object
bf = cv.BFMatcher()

# Match descriptors
matches = bf.knnMatch(desc1, desc2, k=2)

good_matches = []
for m,n in matches:
    if m.distance < .7*n.distance:
        good_matches.append(m)


print(len(good_matches))

# Draw n first matches
img3 = cv.drawMatches(img1, kp1, img2, kp2, good_matches[0:50], None, matchColor=(255,0,0), flags=2)
cv.namedWindow("matches", cv.WINDOW_NORMAL)
#cv.imshow('matches', img3)

#cv.waitKey(0)
cv.destroyAllWindows()


plt.imshow(img3)
plt.savefig('sift_matches.png', dpi=300)
plt.show()


## homography
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

matchesMask = mask.ravel().tolist()
h, w = img1.shape
corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
dst = cv.perspectiveTransform(corners, homography)


img2 = cv.polylines(img2,[np.int32(dst)],True,(255,0,0),3, cv.LINE_AA)
x = int(np.sum(dst[:,0,0]) / 4)
y = int(np.sum(dst[:,0,1]) / 4)
img_pt = cv.circle(img2, (x,y), 15, (255,128,0), thickness=4)


draw_params = dict(matchColor = (255,0,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv.drawMatches(img1,kp1,img2,kp2,good_matches,None,**draw_params)

plt.imshow(img3)
plt.savefig('homo_matches.png', dpi=300)
plt.show()
