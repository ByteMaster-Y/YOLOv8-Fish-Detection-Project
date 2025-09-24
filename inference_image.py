from ultralytics import YOLO

# 학습된 모델 로드 (경로에 맞게 수정)
model = YOLO(r"C:/Users/User/runs/detect/train15/weights/best.pt")

# 이미지 폴더에 대한 예측 수행 및 결과 저장 (conf 임계값 설정)
results = model.predict(source=r"C:/Users/User/your_url/Fish-breeds/test/images", save=True, conf=0.25)

# 결과 하나 확인 (첫 번째 이미지)
results[0].show()

# 여러 결과를 확인하고 싶을 경우
for result in results:
    result.show()