import cv2
import numpy as np
def Object_Tracking():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    measurement = np.array((2, 1), np.float32)
    prediction = np.zeros((2, 1), np.float32)
    input_image_path = 'Input\\Input_Img.png'
    frame = cv2.imread(input_image_path)
    bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
    kalman.statePre = np.array([bbox[0], bbox[1], 0, 0], np.float32)
    kalman.statePost = np.array([bbox[0], bbox[1], 0, 0], np.float32)
    prediction = kalman.predict()
    predicted_bbox = (int(prediction[0]), int(prediction[1]), int(bbox[2]), int(bbox[3]))
    cv2.rectangle(frame, (predicted_bbox[0], predicted_bbox[1]),
                  (predicted_bbox[0] + predicted_bbox[2], predicted_bbox[1] + predicted_bbox[3]),
                  (0, 255, 0), 2)
    cv2.imshow("Object Tracking with Kalman Filter", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    input_img = cv2.imread('Input\\Input_Img.png')
    if input_img is None:
        print("Error: Unable to load the image. Please check the file path.")
        exit(1)
    template = input_img[100:200, 100:200]
    result = cv2.matchTemplate(input_img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    cv2.rectangle(input_img, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imshow('Kernel-Based Tracking', input_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    input_img = cv2.imread('Input\\Input_Img.png')
    if input_img is None:
        print("Error: Unable to load the image. Please check the file path.")
        exit(1)
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    _, binary_silhouette = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Silhouette-Based Tracking', input_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
