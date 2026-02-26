from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO('yolov8n.pt')
    
    model.train(
        data='C:\\Users\\shama\\CV\\hw2\\numberdetection\\data.yaml',
        epochs=50,
        imgsz=640,
        batch=32,
        device=0,
        project='runs/train',
        name='number_detection',
        exist_ok=True,
        optimizer='auto',
    )

