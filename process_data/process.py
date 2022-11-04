import cv2
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation

video_front = '../video/front30.mp4'
video_side = '../video/side30.mp4'

vid_f = cv2.VideoCapture(video_front)
vid_s = cv2.VideoCapture(video_side)

print(vid_f.get(cv2.CAP_PROP_FPS))
print(vid_s.get(cv2.CAP_PROP_FPS))

def resize(image, ratio):
    h,w = image.shape[:2]

    h = int(h*ratio)
    w = int(w*ratio)

    return cv2.resize(image, (w,h))

segmentor = SelfiSegmentation()


while True:
    _, frame_f = vid_f.read()
    _, frame_s = vid_s.read()

    frame_f = segmentor.removeBG(frame_f, (0,0,0),0.6)
    frame_s = segmentor.removeBG(frame_s, (0,0,0), 0.6)

    result = np.vstack([frame_f,frame_s])
    result = resize(result, 0.25)
    cv2.imshow("res", result)
    if cv2.waitKey(100) == ord('q'):
        break