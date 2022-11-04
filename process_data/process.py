import cv2
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
import os

video_front = '../video/front30.mp4'
video_side = '../video/side30.mp4'

output_path = '../data/'
output_f = output_path+'front/'
output_s = output_path+'side/'

os.makedirs(output_f, exist_ok=True)
os.makedirs(output_s, exist_ok=True)

vid_f = cv2.VideoCapture(video_front)
vid_s = cv2.VideoCapture(video_side)

print(vid_f.get(cv2.CAP_PROP_FPS))
print(vid_s.get(cv2.CAP_PROP_FPS))

def resize(image, ratio):
    h,w = image.shape[:2]

    h = int(h*ratio)
    w = int(w*ratio)

    return cv2.resize(image, (w,h))

detector = FaceDetector()
counter = 0

while True:
    _, frame_f = vid_f.read()
    if counter % 100 == 0:
        _, bboxs = detector.findFaces(frame_f.copy())

    center = bboxs[0]["center"]

    _, frame_s = vid_s.read()

    face_f = frame_f[center[1]-200:center[1]+150, center[0]-175:center[0]+175]
    face_f = cv2.resize(face_f, (400,400))

    face_s = frame_s[80:480, 814:1214]

    result = np.vstack([frame_f,frame_s])
    face = np.hstack([face_f,face_s])
    result = resize(result, 0.25)
    cv2.imshow("res", result)
    cv2.imshow('face', face)

    cv2.imwrite(f'{output_f}{counter}.jpg',face_f)
    cv2.imwrite(f'{output_s}{counter}.jpg',face_s)
    counter += 1
    if cv2.waitKey(1) == ord('q'):
        break