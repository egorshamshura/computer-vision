from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n-seg.pt')

    results = model.train(
        data='yolo_dataset/dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project='roadsigns',
        name='road_signs'
    )
