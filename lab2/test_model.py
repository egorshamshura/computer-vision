from ultralytics import YOLO
import cv2

model = YOLO('C:\\Users\\shama\\CV\\hw2\\runs\\detect\\runs\\train\\svhntrained_then_number_detection\\weights\\best.pt')


results = model.predict(
    source="orig_photos",
    conf=0.25,
    iou=0.5,
    save=True,
    imgsz=640,
    device=0 
)
