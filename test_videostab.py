import pyvideostab

import cv2

cap = cv2.VideoCapture('/app/base/video/Auto_stopped_backdriving.mkv')
while True:
    img = cap.read()[1]
    img = cv2.resize(img, dsize=None, fx=.25,fy=.25, interpolation=cv2.INTER_CUBIC)

    cv2.imshow('source', img)
    k = cv2.waitKey(30) & 0xff
    stabilized = pyvideostab.stabilize(img)
    cv2.imshow('stabilized', stabilized)