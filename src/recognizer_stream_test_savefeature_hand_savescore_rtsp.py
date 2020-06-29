import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from imutils import paths
from openpyxl import Workbook
import face_preprocess
import numpy as np
import face_model
import argparse
import pickle
import time
import dlib
import cv2
import os
from src.hand_tracker import HandTracker

write_wb = Workbook()
write_ws = write_wb.active
write_ws.append(["Menu", "Name", "Score"])
WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

POINT_COLOR = (0, 255, 0)
KEY_COLOR =(0,0,255)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

ap = argparse.ArgumentParser()

ap.add_argument("--mymodel", default="outputs/my_model.h5",
    help="Path to recognizer model")
ap.add_argument("--le", default="outputs/le.pickle",
    help="Path to label encoder")
ap.add_argument("--embeddings", default="outputs/embeddings.pickle",
    help='Path to embeddings')
ap.add_argument("--video-out", default="../datasets/videos_output/stream_test.mp4",
    help='Path to output video')


ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='../insightface/models/model-y1-test2/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = ap.parse_args()

# Load embeddings and labels
data = pickle.loads(open(args.embeddings, "rb").read())
#le = pickle.loads(open(args.le, "rb").read())	#200625

embeddings = np.array(data['embeddings'])
print("first:{}".format(np.shape(embeddings)))
labels = np.array(data['names'])        	#200625
#labels = le.fit_transform(data['names'])	
#labels = le.transform(data['names'])	        #200625

indexNumber = 0
#labels = np.array(data['names'])

# Initialize detector
detector = MTCNN()

# Initialize faces embedding model
embedding_model =face_model.FaceModel(args)

# Load the classifier model
model = load_model('outputs/my_model.h5')

#200626 
#def classify(vector1, vector2)
#    vec1 = vector1.flatten()
#    vec2 = vec2.flatten()
#    return False
# Define distance function
def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1, vec2)    #200626   a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    curr = 0
    minimum = 10
    temp = -1
    index = 0
    for source_vec in source_vecs:
        #cos_dist += findCosineDistance(test_vec, source_vec)
        curr = findCosineDistance(test_vec, source_vec)
        if curr < minimum :
            temp = index
            minimum = curr
        cos_dist += curr
        index += 1
    return cos_dist/len(source_vecs), temp

# Initialize some useful arguments
cosine_threshold = 0.97 #0.8
proba_threshold = 0.85
comparing_num = 5
trackers = []
texts = []
handTrackers = []
handTexts = []
frames = 0

draw_points = None
counter = 0

# Start streaming and recording
#rtsp = 'rtsp://admin:**graphics@163.152.162.191:554/cam/realmonitor?channel1&subtype=1'
#cap = cv2.VideoCapture(rtsp)
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_width = 600
save_height = int(600/frame_width*frame_height)
#video_out = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (save_width,save_height))

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16), 
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

detector1 = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

previous_embedding = []
new_labels = []
new_embeddings = []
max_bbox = np.zeros(4)
prev_name = None
name = None
score_number = []
while True:
    ret, frame = cap.read()
    frames += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (save_width, save_height))
    detect_tock=time.time()
    detect_tick=time.time()
    cnt=[]
    name = None
    #detect_tick = time.time()
    bboxes = detector.detect_faces(frame)
    #detect_tock = time.time()
    if len(bboxes) != 0:
        reco_tick = time.time()
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

        max_bbox = max_bbox[0:4]
        if frames%2 == 0:
            trackers = []
            texts = []
            handTrackers = []
            handTexts = []

            counter = 0

            landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                 landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
            landmarks = landmarks.reshape((2,5)).T
            nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            nimg = np.transpose(nimg, (2,0,1))
            embedding = embedding_model.get_feature(nimg).reshape(1,-1)

            text = "Unknown"

            ### Predict class
            #preds = model.predict(embedding)
            #preds = preds.flatten()

            ## Get the highest accuracy embedded vector
            #j = np.argmax(preds)
            #proba = preds[j]

            ## Compare this vector to source class vectors to verify it is actual belong to this class
            #match_class_idx = (labels == j)
            #match_class_idx = np.where(match_class_idx)[0]
            #selected_idx = np.random.choice(match_class_idx, comparing_num)
            #compare_embeddings = embeddings[selected_idx]

            # Calculate cosine similarity
            cos_similarity, j = CosineSimilarity(embedding, embeddings)
            # print("Sim:{:.2f}, index:{}".format(cos_similarity,j))

            if previous_embedding == []:
                previous_cos_similarity = cosine_threshold
                print("!!!first")
            else:
                previous_cos_similarity, _ = CosineSimilarity(embedding, previous_embedding)
            if cos_similarity < cosine_threshold: #and proba > proba_threshold:
                # hand detection
                #points, hand_box, land_points = detector1(rgb)
                draw_points, points,hand_box = detector1(rgb)
                print("I know you and your name")
                #name = le.classes_[j]#200625
                name = labels[j]
                print("name : {}".format(name))
                #name = labels[j]
                text = "{}".format(name)
                #print("Recognized: {} <{:.2f}>".format(name, proba*100))
                if draw_points is not None:
                    for point in draw_points:
                        x, y = point
                        cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
                    for connection in connections:
                        x0, y0 = draw_points[connection[0]]
                        x1, y1 = draw_points[connection[1]]
                        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
                    for i in range(0, 4):
                        x0, y0 = hand_box[i]
                        if i != 3:
                            x1, y1 = hand_box[i+1]
                        else :
                            x1, y1 = hand_box[0]
                        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), THICKNESS)
                    # for keypoint in seven_kps:
                    #     x,y = keypoint
                    #     cv2.circle(frame,(int(x), int(y)), 10, KEY_COLOR, 5)
                    # cv2.rectangle(frame, (int(hand_box[0][0]), int(hand_box[0][1])),
                    #               (int(hand_box[2][0]), int(hand_box[2][1])), (255, 255, 255), 2)
                    thumbState = 'UNKNOW'
                    firstFingerState = 'UNKNOW'
                    secondFingerState = 'UNKNOW'
                    fourthFingerState = 'UNKNOW'
                    IsLefthand = 'UNKNOW'
                    IsRighthand = 'UNKNOW'

                    if (points[5][0] < points[17][0]):
                        IsLefthand = True
                    else:
                        IsLefthand = False

                    land2 = points[2][0]
                    land3 = points[3][0]
                    land4 = points[4][0]

                    if (IsLefthand):
                        land2 *= -1.0
                        land3 *= -1.0
                        land4 *= -1.0
                    pseudoFixKeyPoint = land2
                    if (land3 < pseudoFixKeyPoint and land4 < pseudoFixKeyPoint):
                        thumbState = 'CLOSE'
                    elif (pseudoFixKeyPoint < land3 and pseudoFixKeyPoint < land4):
                        thumbState = 'OPEN'
                        counter += 1
                    pseudoFixKeyPoint = points[6][1]
                    if (points[7][1] < pseudoFixKeyPoint and points[8][1] < points[7][1]):
                        thumbState = 'OPEN'
                        counter += 1
                    elif (pseudoFixKeyPoint < points[7][1] and points[7][1] < points[8][1]):
                        thumbState = 'CLOSE'

                    pseudoFixKeyPoint = points[10][1]
                    if (points[11][1] < pseudoFixKeyPoint and points[12][1] < points[11][1]):
                        thumbState = 'OPEN'
                        counter += 1
                    elif (pseudoFixKeyPoint < points[11][1] and points[11][1] < points[12][1]):
                        thumbState = 'CLOSE'

                    pseudoFixKeyPoint = points[14][1]
                    if (points[15][1] < pseudoFixKeyPoint and points[16][1] < points[15][1]):
                        thumbState = 'OPEN'
                        counter += 1
                    elif (pseudoFixKeyPoint < points[15][1] and points[15][1] < points[16][1]):
                        thumbState = 'CLOSE'

                    pseudoFixKeyPoint = points[18][1]
                    if (points[19][1] < pseudoFixKeyPoint and points[20][1] < points[19][1]):
                        thumbState = 'OPEN'
                        counter += 1
                    elif (pseudoFixKeyPoint < points[19][1] and points[19][1] < points[20][1]):
                        thumbState = 'CLOSE'

                if counter != 0:
                    strCnt = str(counter)
                    cnt.append(counter)
                    # cx1, cy1 = point[0]
                    cv2.putText(frame, strCnt, (int(max_bbox[0]), int(y + 5)), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                (255, 0, 255), 2)
                    # cv2.putText(frame, strCnt, (int(cx1), int(cy1)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255),2)
                    # print("new score: %d " % (cnt[0]))

            else:
                # print("I don't know you")
                if previous_cos_similarity >= cosine_threshold:
                    # print("Unknown_new_face")
                    text = "Person" + str(indexNumber)
                    #new_embeddings.append(embedding)
                    #new_labels.append(text + str(indexNumber))
                    embeddings=np.concatenate((embeddings, embedding))
                    labels=np.append(labels, text)
                    indexNumber += 1
                    previous_embedding.append(embedding)
                    new_labels.append(text)
                    new_embeddings.append(embedding)
                else:
                    _, k = CosineSimilarity(embedding, previous_embedding)
                    name = new_labels[k]
                    # print("name : {}".format(name))
                    text = "{}".format(name)



            # Start tracking
            tracker = dlib.correlation_tracker()
            tx0, ty0, tx1, ty1 = (max_bbox[0], max_bbox[1], max_bbox[2], max_bbox[3])
            rect = dlib.rectangle(int(tx0), int(ty0), int(tx1), int(ty1))
            #rect = dlib.rectangle(max_bbox[0], max_bbox[1], max_bbox[2], max_bbox[3])
            tracker.start_track(rgb, rect)
            trackers.append(tracker)
            texts.append(text)

            y = max_bbox[1] - 10 if max_bbox[1] - 10 > 10 else max_bbox[1] + 10
            # print("max_bbox : %.2f" %(max_bbox[0]))
            # print("y: %.2f" %(y))
            cv2.putText(frame, text, (max_bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), (255,0,0), 2)


        else:
            for tracker, text in zip(trackers,texts):
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                cv2.rectangle(frame, (startX, startY), (endX, endY), (255,0,0), 2)
                cv2.putText(frame, text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                if draw_points is not None:
                    for point in draw_points:
                        x, y = point
                        cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
                    for connection in connections:
                        x0, y0 = draw_points[connection[0]]
                        x1, y1 = draw_points[connection[1]]
                        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
                if counter != 0:
                    strCnt = str(counter)

                    cv2.putText(frame, strCnt, (int(max_bbox[0]), int(y+5)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255),2)
                

    cv2.imshow("Frame", frame)
    # video_out.write(frame)
    #print("Faces detection time: {}s".format(detect_tock-detect_tick))
    # print("Faces recognition time: {}s".format(reco_tock-reco_tick))
    key = cv2.waitKey(1) & 0xFF
    if name != None and counter != 0:

        score_number.append(counter)

        if prev_name != name and prev_name != None:
        #write_ws.append([1, name, counter])

            id = len(score_number) -1
            if(id >= 3):
                for x in range(3, id):
                    curr_score = score_number[x-1]
                    pre_score = score_number[x-2]
                    prepre_score = score_number[x-3]
                    if(curr_score == pre_score & pre_score == prepre_score):
                        final_score = curr_score
                        write_ws.append([1, prev_name, final_score])
                        break

            first_score = score_number[id]
            score_number = []
            print(score_number)
            score_number.append(first_score)
            print(score_number)
        prev_name = name
    if key == ord("q"):
        break

if name != None and counter != 0:
    id = len(score_number) -1
    if(id >= 3):
        for x in range(3, id):
            curr_score = score_number[x-1]
            pre_score = score_number[x-2]
            prepre_score = score_number[x-3]
            if(curr_score == pre_score & pre_score == prepre_score):
                final_score = curr_score
                write_ws.append([1, name, final_score])
                break
# video_out.release()
cap.release()
#save to output
print(np.shape(embeddings))
new_data = {"embeddings": embeddings, "names": labels}
f = open(args.embeddings, "wb")
f.write(pickle.dumps(new_data))
f.close()
write_wb.save('excel_test.xlsx')
cv2.destroyAllWindows()
