from ultralytics import YOLO 
import cv2  

# 모델 로드 (경로에 맞게 수정)
# 미리 학습된 YOLOv8 모델의 가중치 파일(best.pt)을 불러옵니다. 이 파일은 학습 후 runs/detect 폴더에 저장됩니다.
model = YOLO(r"C:\Users\User\runs\detect\train15\weights\best.pt")

# 비디오 파일 열기 (경로에 맞게 수정)
# cv2.VideoCapture() 함수를 사용해 동영상 파일을 읽어옵니다.
cap = cv2.VideoCapture(r"C:\Users\User\your_url\test_video\fish2.mp4")
frameCount = 0  # 현재 처리 중인 프레임 수를 저장할 변수 초기화

while (True):  # 비디오의 모든 프레임을 반복 처리하기 위한 무한 루프
    ret, frame = cap.read()  # 비디오에서 다음 프레임을 읽어옵니다.
                             # ret는 프레임을 성공적으로 읽었는지(True/False)를, frame은 실제 프레임 데이터를 저장합니다.
    if (not(ret)):  # 프레임을 더 이상 읽을 수 없으면 (비디오 끝에 도달)
        break  # 루프를 종료합니다.
    
    # 프레임 크기 조절
    # cv2.resize() 함수로 프레임의 크기를 640x360으로 변경합니다. 이는 모델 입력 크기를 맞추고 처리 속도를 높이는 데 유용합니다.
    frame = cv2.resize(frame, dsize=(640, 360))
    
    # YOLOv8 모델로 프레임에 대한 예측 수행
    # model.predict() 함수를 사용하여 현재 프레임(frame)에서 객체를 탐지합니다.
    # source=frame: 입력 소스를 현재 프레임으로 지정합니다.
    # show=True: (선택사항) 예측 결과를 새로운 창에 자동으로 표시합니다.
    # verbose=False: 추론 과정의 상세한 로그를 출력하지 않습니다.
    # stream=False: 프레임 단위로 결과를 처리합니다 (True로 설정 시 스트림 모드).
    # conf=0.7: 신뢰도(Confidence) 임계값을 0.7로 설정합니다. 이 값보다 낮은 예측은 무시합니다.
    # imgsz=640: 모델이 입력받을 이미지의 크기를 640x640으로 지정합니다.
    results = model.predict(source=frame, show=True, verbose=False, stream=False, conf=0.7, imgsz=640)
    res = results[0]  # predict 결과는 리스트 형태이므로, 첫 번째 프레임의 결과만 가져옵니다.

    # 결과에서 Bounding Box 정보 추출 및 시각화
    for box in res.boxes:  # 탐지된 모든 객체(Bounding Box)에 대해 반복합니다.
        # print(f"FrameCount = {frameCount}, {box.data.cpu().numpy()}")
        
        # Bounding Box 좌표와 클래스 ID
        # box.xyxy: Bounding Box의 (x1, y1, x2, y2) 좌표를 가져옵니다. cpu().numpy()로 numpy 배열로 변환합니다.
        npp = box.xyxy.cpu().numpy()
        # box.cls: 탐지된 객체의 클래스 ID를 가져옵니다.
        npcls = box.cls.cpu().numpy()
        
        # 중심 좌표 계산
        # Bounding Box의 좌측 상단(npp[0][0], npp[0][1])과 우측 하단(npp[0][2], npp[0][3]) 좌표를 이용해 중심점을 계산합니다.
        cx = int((npp[0][0] + npp[0][2]) / 2)
        cy = int((npp[0][1] + npp[0][3]) / 2)
        
        # 중심점 그리기
        # cv2.circle() 함수를 사용해 계산된 중심 좌표에 원을 그립니다.
        frame = cv2.circle(frame, (cx, cy), 30, (0, 255, 255), 3)
        
        # 클래스 이름 표시
        # 탐지된 객체의 클래스 ID를 정수형으로 변환합니다.
        class_id = int(npcls[0])
        # res.names: 모델의 클래스 이름 딕셔너리에서 ID에 해당하는 이름을 가져옵니다.
        class_name = res.names[class_id]
        # cv2.putText() 함수로 프레임 위에 클래스 이름을 텍스트로 표시합니다.
        cv2.putText(
            frame,
            class_name,
            org=(cx, cy),  # 텍스트를 그릴 위치 (중심점)
            color=(0, 255, 0),  # 텍스트 색상 (녹색)
            fontScale=1,  # 폰트 크기
            thickness=2,  # 폰트 두께
            lineType=cv2.LINE_AA,  # 선 타입
            fontFace=cv2.FONT_HERSHEY_SIMPLEX  # 폰트 종류
        )
    
    # 결과 화면에 출력
    # cv2.imshow() 함수로 처리된 프레임을 'Detected Object'라는 이름의 창에 보여줍니다.
    cv2.imshow('Detected Object', frame)
    
    # ESC 키 누르면 종료
    # cv2.waitKey(1)은 1ms 동안 키 입력을 기다립니다. 27은 ESC 키의 아스키 코드입니다.
    if (cv2.waitKey(1) == 27):
        break  # ESC 키가 눌리면 루프를 종료합니다.
    
    frameCount += 1  # 다음 프레임을 위해 프레임 카운트 증가

# 리소스 해제
# cap.release()와 cv2.destroyAllWindows()는 사용이 끝난 비디오 파일과 창을 닫아 리소스를 해제합니다.
cap.release()
cv2.destroyAllWindows()