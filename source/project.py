# -*- coding: utf-8 -*-
#
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import tensorflow as tf
import keras

from keras.models import load_model
model = load_model('./model/05-0.0488.hdf5')
model.summary()



def ImageProcessing(Img):
    grayImg = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(grayImg, (5,5), 2)

    kernel = np.ones((10,10), np.uint8)
    morphImg =cv2.morphologyEx(blurImg, cv2.MORPH_OPEN, kernel)

    ret, threImg = cv2.threshold(morphImg, 150, 230, cv2.THRESH_BINARY_INV)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        image, contours, hierachy = cv2.findContours(threImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierachy = cv2.findContours(threImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(each) for each in contours]

    ImgResult = []
    Img_for_class = Img.copy()
    pixel = 0
    for rect in rects:
        target_num = Img_for_class[rect[1] - pixel: rect[1] + rect[3] + pixel,rect[0] - pixel: rect[0] + rect[2] + pixel]
        test_num = cv2.resize(target_num, (28, 28))[:, :, 1]
        test_num = (test_num < 70) * 255
        test_num = test_num.astype('float32') / 255.
        #lt.imshow(test_num, cmap='gray', interpolation='nearest')
        test_num = test_num.reshape((1, 28, 28, 1))
        predictNum = model.predict_classes(test_num)
         # Draw the rectangles
        cv2.rectangle(Img, (rect[0], rect[1]),(rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(Img, str(predictNum[0]), (rect[0], rect[1]), font, 1, (255, 0, ), 3)

    return Img



#####################################################
capture = cv2.VideoCapture(0)

if capture.isOpened():
    print("Video Opened")
else:
    print("Video Not Opened")
    print("Program Abort")
    exit()
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)
    output = ImageProcessing(frame)
    cv2.imshow("Output", output)
    if cv2.waitKey(1) > 0: break

capture.release()
cv2.destroyAllWindows()
