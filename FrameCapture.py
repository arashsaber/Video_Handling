import numpy as np
import cv2


import sys
sys.path.append('/home/arash/Desktop/models/research/object_detection')
from utils import label_map_util
from utils import visualization_utils as vis_util
#   ---------------------------------------
video_file = './data/city.mp4'
cap = cv2.VideoCapture(video_file)

while(cap.isOpened()):
    ret, frame = cap.read()
    # frame index: cap.get(1)
    # frame w and h: cap.get(3), cap.get(4)
    print(cap.get(0), cap.get(1), cap.get(2), cap.get(3), cap.get(4))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()