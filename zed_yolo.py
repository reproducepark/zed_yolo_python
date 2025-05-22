import pyzed.sl as sl
import cv2
import time
from collections import deque
from ultralytics import YOLO
import numpy as np

# 1. ZED 카메라 초기화
zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD720
init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
status = zed.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print("ZED 카메라 열기 실패")
    exit(1)

runtime = sl.RuntimeParameters()
image = sl.Mat()
depth = sl.Mat()

# 2. YOLO11 모델 로드
model = YOLO('yolo11n.pt')  # 또는 사용자 정의 모델
model.to('cuda')

# FPS 계산용 변수
prev_time = time.time()
frame_times = deque(maxlen=30)  # 최근 30프레임 기준 평균 FPS

# 3. 메인 루프
while True:
    current_time = time.time()
    elapsed = current_time - prev_time
    prev_time = current_time

    frame_times.append(elapsed)
    avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        frame = image.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        # YOLO 추론
        results = model(frame, device='cuda')[0]  # <-- 매 추론마다 명시
        print("YOLO device:", model.device)

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            points = [
                (x1, y1), ((x1 + x2)//2, y1), (x2, y1),
                (x1, (y1 + y2)//2), (cx, cy), (x2, (y1 + y2)//2),
                (x1, y2), ((x1 + x2)//2, y2), (x2, y2),
            ]

            depths = []
            for px, py in points:
                if 0 <= px < depth.get_width() and 0 <= py < depth.get_height():
                    status, depth_val_mm = depth.get_value(px, py)
                    if status == sl.ERROR_CODE.SUCCESS and depth_val_mm > 0:
                        depths.append(depth_val_mm / 1000.0)  # mm → m

            if depths:
                median_depth = np.median(depths)
                label = f"{median_depth:.3f} m"
            else:
                label = "No depth"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("ZED + YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

zed.close()
