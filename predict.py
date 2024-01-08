import os

from ultralytics import YOLO

import cv2
import time
#when the y in the image is more high its more small in the code (like arkanoid)
def bounce(posList, frame):
    if (len(posList) < 3):
        return
    if posList[-1][1] < posList[-2][1] and posList[-2][1] > posList[-3][1]:

        posList[-2] = (posList[-2][0],posList[-2][1],True)


VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'v3.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
#33 is the best
# 7 is where the table is good
# now trying 9 because it finishes 800 runs
model_path = os.path.join('.', 'runs', 'detect', 'train8', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.00001
posList = []
while ret:
#max_det=3  in model.predict say how many object i want to detect
    results = model.predict(frame, )[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        xCenter = (int)((int(x1) + int(x2)) / 2)
        yCenter = (int)((int(y1) + int(y2)) / 2)

        if score > threshold:
            if (class_id == 0):
                posList.append((xCenter, yCenter, False))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA, )
            if len(posList) > 50:
                posList.pop(0)
            bounce(posList, frame)
            for i, pos in enumerate(posList):
                if pos[2]:
                    cv2.circle(frame, (pos[0], pos[1]), 15, (0, 0, 255), cv2.FILLED)
                else:
                    cv2.circle(frame, (pos[0], pos[1]), 5, (0, 255, 0), cv2.FILLED)
                if i != 0:
                    cv2.line(frame, (pos[0], pos[1]), (posList[i - 1][0], posList[i - 1][1]), (0, 0, 255), 2)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()











