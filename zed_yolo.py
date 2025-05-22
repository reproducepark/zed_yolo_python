import pyzed.sl as sl
import cv2
import time
import numpy as np
from collections import deque
from ultralytics import YOLO


def init_zed():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("ZED 카메라 열기 실패")
        exit(1)

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    return zed, runtime, image, depth


def load_yolo_model(weights_path='yolo11n.pt', device='cuda'):
    model = YOLO(weights_path)
    model.to(device)
    return model


def get_median_depth(depth_map, x1, y1, x2, y2):
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    points = [
        (x1, y1), ((x1 + x2)//2, y1), (x2, y1),
        (x1, (y1 + y2)//2), (cx, cy), (x2, (y1 + y2)//2),
        (x1, y2), ((x1 + x2)//2, y2), (x2, y2),
    ]

    depths = []
    for px, py in points:
        if 0 <= px < depth_map.get_width() and 0 <= py < depth_map.get_height():
            status, depth_val_mm = depth_map.get_value(px, py)
            if status == sl.ERROR_CODE.SUCCESS and depth_val_mm > 0:
                depths.append(depth_val_mm / 1000.0)

    return np.median(depths) if depths else None


def process_frame(frame, depth_map, results):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        median_depth = get_median_depth(depth_map, x1, y1, x2, y2)
        label = f"{median_depth:.3f} m" if median_depth else "No depth"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def main():
    zed, runtime, image, depth = init_zed()
    model = load_yolo_model()
    frame_times = deque(maxlen=30)
    prev_time = time.time()

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

            results = model(frame, device='cuda')[0]
            process_frame(frame, depth, results)

            cv2.putText(frame, f"Avg FPS: {avg_fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("ZED + YOLO Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()


if __name__ == "__main__":
    main()
