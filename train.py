# 모델 학습 윈도우 기준
from ultralytics import YOLO

# --- 학습 설정 1: 기본 (epochs=15) ---
model = YOLO(r"C:\Users\User\your_url\yolov8n.pt")
result = model.train(data=r"C:\Users\User\your_url\data.yaml", epochs=15)


# --- 학습 설정 2: scratch (epochs=30, batch=16) ---
model = YOLO(r"C:\Users\User\your_url\yolov8n.pt")
result = model.train(data='data.yaml', epochs=30, batch=16)


# --- 학습 설정 3: freeze (epochs=30, freeze=10, batch=16) ---
# 모델의 첫 10개 레이어를 고정하고 학습
model = YOLO(r"C:\Users\User\your_url\yolov8n.pt")
result = model.train(data=r"C:\Users\User\your_url\data.yaml", epochs=30, freeze=10, batch=16)