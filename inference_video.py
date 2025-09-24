from ultralytics import YOLO

# 학습된 모델 로드 (경로에 맞게 수정)
model = YOLO(r"C:/Users/User/runs/detect/train15/weights/best.pt")

# 비디오 파일 경로
video_path = r"C:\Users\User\your_url\test_video\fish2.mp4"

# 비디오에 대한 예측 수행 및 결과 저장
results = model.predict(source=video_path, save=True, conf=0.25)