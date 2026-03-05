import os
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def track_video(video_path, model_path, tracker_config, output_dir):
    model = YOLO(model_path)
    video_name = Path(video_path).stem
    out_video = os.path.join(output_dir, f"{video_name}_{tracker_config}.mp4")
    out_txt = os.path.join(output_dir, f"{video_name}_{tracker_config}.txt")

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    frame_id = 0
    track_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, tracker=tracker_config, persist=True, verbose=False)
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, conf, cls_id in zip(boxes, track_ids, confs, cls_ids):
                x1, y1, x2, y2 = box
                track_log.append([frame_id, tid, x1, y1, x2, y2, conf, cls_id, -1, -1])

        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open(out_txt, 'w') as f:
        f.write("frame,id,x1,y1,x2,y2,conf,class,-1,-1\n")
        for row in track_log:
            f.write(','.join(map(str, row)) + '\n')

    return track_log

if __name__ == '__main__':
    video_path = "C:\\Users\\shama\\CV\\hw3\\IMG_4330.mov"
    model_path = "C:\\Users\\shama\\CV\\hw3\\runs\\segment\\runs\\segment\\road_signs4\\weights\\best.pt"
    output_dir = "tracking_results"
    os.makedirs(output_dir, exist_ok=True)

    trackers = ['botsort.yaml', 'bytetrack.yaml']

    for tracker in trackers:
        print(f"\n---{tracker} ---")
        log = track_video(video_path, model_path, tracker, output_dir)
