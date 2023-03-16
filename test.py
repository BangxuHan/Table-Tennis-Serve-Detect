import numpy as np
import matplotlib.pyplot as plt
import pylab
# import imageio
# import skimage.io
import numpy as np
import cv2

# cap = cv2.VideoCapture('WTT1.mp4')

video_path = 'WTT1.mp4'
img_path = "output"
cap = cv2.VideoCapture()
cap.open(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps, width, height)
suc = cap.isOpened()
print(suc)
frame_count = 0
while suc:
    frame_count += 1
    suc, frame = cap.read()
    cv2.imwrite(img_path + str(frame_count) + '.png', frame)
    cv2.waitKey(1)
cap.release()
print('End!')

# import cv2
# print(cv2.getBuildInformation())
