import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
def preprocess():
    image_path = "Input\\Input_Img.png"
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Edge Detection')
    plt.show()
    output_image_path = "Output\\PreprocessingImg.png"
    cv2.imwrite(output_image_path, img)
    print(f"Preprocessing image saved at: {output_image_path}")
    image_path = "Input\\Input_Img.png"
    img = cv2.imread(image_path)
    mean_color = np.mean(img, axis=(0, 1))
    std_color = np.std(img, axis=(0, 1))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray_blurred, 100, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = [cv2.contourArea(cnt) for cnt in contours]
    max_contour_area = max(contour_areas, default=1)
    aspect_ratio = img.shape[1] / img.shape[0]
    feature_vector = np.concatenate([mean_color, std_color, [max_contour_area, aspect_ratio]])
    scaler = StandardScaler()
    feature_vector_standardized = scaler.fit_transform(feature_vector.reshape(1, -1))
    print("Combined Feature Vector:", feature_vector_standardized.flatten())
