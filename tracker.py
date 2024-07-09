import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import argparse
    
model = YOLO('yolov5s.pt')

tracker = DeepSort(max_age=5)

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

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('-p', '--path')
group.add_argument('-n', '--num', type=int)
args = parser.parse_args()

def video_stream_detection(args):

    if args.path:
        if not os.path.isfile(args.path):
            raise FileNotFoundError(f'File on path {args.path} not found')
        cap = cv2.VideoCapture(args.path)
    elif args.num is not None:
        cap = cv2.VideoCapture(args.num)
        if not cap.isOpened():
            raise ValueError(f'Camera with number {args.num} not found. Make sure it exists and is connected.')
    else:
        raise ValueError('Either -p/--path or -n/--num must be specified')
    
    return cap

cap = video_stream_detection(args)

classes_file_path = 'classes.txt'
classes = load_classes(classes_file_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_objects(frame)
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:

        if not track.is_confirmed():
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