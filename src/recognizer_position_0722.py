import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from imutils import paths
from openpyxl import Workbook
from src.hand_tracker_0722 import HandTracker
import face_preprocess
import numpy as np
import face_model
import argparse
import pickle
import time
import dlib
import cv2
import os

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

POINT_COLOR = (0, 255, 0)
KEY_COLOR =(0,0,255)
CONNECTION_COLOR = (255, 0, 0)
NEWFACE_COLOR = (255, 255, 255)
FACEBOX_COLOR=(255,0,0)
THICKNESS = 2

ap = argparse.ArgumentParser()

ap.add_argument("--mymodel", default="outputs/my_model.h5",
    help="Path to recognizer model")
ap.add_argument("--le", default="outputs/le.pickle",
    help="Path to label encoder")
ap.add_argument("--embeddings", default="outputs/embeddings.pickle",
    help='Path to embeddings')


ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='../insightface/models/model-y1-test2/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=1, type=int, help='gpu id')
ap.add_argument('--det', default=1, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = ap.parse_args()

# Initialize detector
detector = MTCNN()

def getMaxfacebox(bboxes):
    max_area = 0
    for bboxe in bboxes:
        bbox = bboxe["box"]
        bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        keypoints = bboxe["keypoints"]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area > max_area:
            max_bbox = bbox
            landmarks = keypoints
            max_area = area
    return max_bbox[0:4], landmarks, max_area


#hand detector
hand_detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

# Initialize some useful arguments
trackers = []
frames = 0
hand_box = None
max_bbox = None

# Start streaming and recording
cap = cv2.VideoCapture(0)

while True:
    # ret is fading
    ret, frame = cap.read()
    frames += 1
    if frames % 3 == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cnt = []
        max_bbox = None
        bboxes_Out = detector.detect_faces(frame)
        draw_points, points, hand_box, hand_area = hand_detector(rgb)
        #trackers = []
        
        #face size check
        if len(bboxes_Out) != 0:
            #reco_tick = time.time()
            max_bbox, landmarks, max_area = getMaxfacebox(bboxes_Out)
            if max_area >= 8000:
                print("Correct position - face area : %d"%(max_area))
                cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), FACEBOX_COLOR, 2)
            else:
                max_bbox = None

        #hand size check
        if hand_box is not None:
            # draw hand box
            for i in range(0, 4):
                x0, y0 = hand_box[i]
                if i != 3:
                    x1, y1 = hand_box[i + 1]
                else:
                    x1, y1 = hand_box[0]
                cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), THICKNESS)
            if hand_area >= 300:
                print("Correct position - hand area : %d"%(hand_area))
    else:
        if max_bbox is not None:
            cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), FACEBOX_COLOR, 2)

        if hand_box is not None:
            for i in range(0, 4):
                x0, y0 = hand_box[i]
                if i != 3:
                    x1, y1 = hand_box[i + 1]
                else:
                    x1, y1 = hand_box[0]
                cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), THICKNESS)
                
    cv2.namedWindow("Output_cam")
    cv2.moveWindow("Output_cam", 840, 30)
    #frame = cv2.resize(frame, (320, 240))
    cv2.imshow("Output_cam", frame)
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
