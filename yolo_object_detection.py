import cv2
from ultralytics import YOLO


model = YOLO('yolov8s.pt')

video_cap = cv2.VideoCapture(0)

while True:
    ret, frame = video_cap.read()
    if not ret:
        continue
    
    results = model.track(frame, stream=True)
    for result in results:
        classes_names = result.names

        for box in result.boxes:
            if box.conf[0] > 0.4:
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = box.cls[0]
                class_name = classes_names[int(cls)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()