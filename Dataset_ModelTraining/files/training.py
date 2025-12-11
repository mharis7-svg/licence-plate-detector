from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data=r"C:\Users\Haris\Desktop\YALPR\Dataset_ModelTraining\files\license-data.yaml",
    epochs=1
)
print("Training complete. Results:", results)
