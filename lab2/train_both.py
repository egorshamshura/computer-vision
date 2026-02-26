from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO('runs/detect/runs/train/number_detection/weights/best.pt')

    model.train(
        data='C:\\Users\\shama\\CV\\hw2\\SVHN\\data.yaml',
        epochs=10,
        imgsz=640,
        batch=32,
        device=0,
        project='runs/train',
        name='svhntrained_then_number_detection',
        exist_ok=True,
        optimizer='auto',
    )