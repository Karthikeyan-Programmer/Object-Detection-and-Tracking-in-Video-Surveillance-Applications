import cv2
import numpy as np
import matplotlib.pyplot as plt
def Object_Detection():
    image_path = "Input\\Input_Img.png"
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    num_frames = 150
    frames = [gray] * num_frames
    output_path = "Output/"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path + "output_video.mp4", fourcc, 10.0, (img.shape[1], img.shape[0]))
    for i in range(1, num_frames):
        frame_diff = cv2.absdiff(frames[i - 1], frames[i])
        flow = cv2.calcOpticalFlowFarneback(
            frames[i - 1], frames[i], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        cv2.imshow("Frame Differencing", frame_diff)
        cv2.imshow("Optical Flow", mag)
        cv2.imshow("Background Subtraction", thresh)
        output_video.write(frame_diff)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    output_video.release()
    cv2.destroyAllWindows()
