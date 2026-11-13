from ultralytics import YOLO




def train_model(data_path, model_path, epochs=100, imgsz=640, batch=2,
                lr0=0.001):
    model = YOLO(model_path)

    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        val=True,
        device = 0,
        batch = batch,
        amp = True,
        lr0 = lr0
    )

    return results


if __name__ == "__main__":
    train_model(
        data_path="https://platform.ultralytics.com/spider-sparrow/datasets/recyle",
        model_path="runs/detect/train-9/weights/best.pt",
    )
