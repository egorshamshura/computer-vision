from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO('yolov8n.pt')
    
    model.train(
        data='C:\\Users\\shama\\CV\\hw2\\SVHN\\data.yaml',
        epochs=10,
        imgsz=640,
        batch=32,
        device=0,
        project='runs/train',
        name='svhntrained_number_detection',
        exist_ok=True,
        optimizer='auto',
    )

