import numpy as np
import cv2
import datetime

def CNN():
    video_cap = cv2.VideoCapture("Input/Input_Video.mp4")
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter("Output\\output.mp4", fourcc, fps, (frame_width, frame_height))
    weights = "ssd_mobilenet/frozen_inference_graph.pb"
    model = "ssd_mobilenet/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(weights, model)
    class_names = []
    with open("ssd_mobilenet/coco_names.txt", "r") as f:
        class_names = f.read().strip().split("\n")
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3))
    while True:
        start = datetime.datetime.now()
        success, frame = video_cap.read()
        h = frame.shape[0]
        w = frame.shape[1]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0/127.5, (320, 320), [127.5, 127.5, 127.5])
        net.setInput(blob)
        output = net.forward() 
        for detection in output[0, 0, :, :]: 
            probability = detection[2]
            if probability < 0.5:
                continue
            class_id = int(detection[1])
            label = class_names[class_id - 1].upper()
            color = colors[class_id]
            B, G, R = int(color[0]), int(color[1]), int(color[2])
            box = [int(a * b) for a, b in zip(detection[3:7], [w, h, w, h])]
            box = tuple(box)
            cv2.rectangle(frame, box[:2], box[2:], (B, G, R), thickness=2)

            text = f"{label} {probability * 100:.2f}%"
            cv2.putText(frame, text, (box[0], box[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        end = datetime.datetime.now()
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
        cv2.imshow("Output", frame)
        writer.write(frame)
        if cv2.waitKey(10) == ord("q"):
            break
    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()
