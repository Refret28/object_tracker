import cv2
from ultralytics import YOLO
# import sys
# sys.path.append('deep_sort_realtime') # if there is an error importing a module, add the path to find it
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

model = YOLO('yolov5s.pt')

tracker = DeepSort(max_age=30)

def load_classes(file_path):
    with open(file_path, 'r') as f:
        classes = f.read().strip().split('\n')
    return classes

def detect_objects(frame):
    results = model(frame)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            class_id = int(cls)
            if class_id < len(classes):  
                class_name = classes[class_id]  
                if class_name in classes:  
                    detections.append(([x1.item(), y1.item(), (x2 - x1).item(), (y2 - y1).item()], conf.item(), class_name))
    return detections

#video_path = os.path.join('...') # path to video stream
cap = cv2.VideoCapture(0) # you can specify the camera number

classes_file_path = 'classes.txt'
classes = load_classes(classes_file_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_objects(frame)
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:

        if track.is_confirmed() is False:
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        left, top, right, bottom = map(int, ltrb)  
        class_ = track.det_class
        class_name = class_ if class_ else 'Unknown'

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
        cv2.putText(frame, f'id: {track_id}, class: {class_name}', (left, top-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 0), 2)

    cv2.imshow('Object tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()