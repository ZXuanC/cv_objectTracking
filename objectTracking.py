import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

cap = cv2.VideoCapture(0)

width = 1280
height = 960

cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

lower_skin = np.array([0,58,40])
upper_skin = np.asarray([35,174,255])

#initialize
area = width * height
ret, frame = cap.read()
avg = cv2.blur(frame,(4,4))
avg_float = np.float32(avg)

while True:
    ret,frame = cap.read()
    cv2.imshow('Cap',frame)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    thresh = cv2.inRange(hsv,lower_skin,upper_skin)

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contoursImg, contours , _ = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # 忽略太小的區域
        if cv2.contourArea(c) < 1000:
            continue

        # 偵測到物體，可以自己加上處理的程式碼在這裡...

        # 計算等高線的外框範圍
        (x, y, w, h) = cv2.boundingRect(c)

        # 畫出外框
        cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)

        # 畫出等高線（除錯用

    cv2.imshow('thresh',thresh)
    res=cv2.bitwise_and(frame,frame,mask=thresh)
    cv2.imshow('res',res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.waitKey(0)

