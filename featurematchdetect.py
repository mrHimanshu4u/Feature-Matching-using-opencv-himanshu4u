import cv2
import numpy as np



cap = cv2.VideoCapture(0)
img1 = cv2.imread('thinkmonkcover.jpg')

orb = cv2.ORB_create(nfeatures=1450)


while True:

    success , img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)


    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.78 * n.distance:
            good.append([m])
        a = len(good)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    if a > 40:
        cv2.putText(imgOriginal, 'thinkmonk', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),2)
    print(a)
    cv2.imshow('img2', imgOriginal)
    cv2.waitKey(1)

