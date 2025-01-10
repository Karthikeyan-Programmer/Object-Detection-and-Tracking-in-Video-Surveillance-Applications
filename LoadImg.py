import os
from pathlib import Path
import shutil
import time
from tkinter import filedialog
from plyer import notification
from PIL import Image
import cv2
def display_toast(title, message):
    max_title_length = 64
    truncated_title = title[:max_title_length]
    notification.notify(
        title=truncated_title,
        message=message,
        timeout=5
    )
def Load():
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",
                  'Load KITTI DATASET IMAGE')
    print("\nSelect KITTI Dataset Image File:")
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ["*.png", "*.jpg"]),
                                                       ("Video Files", ["*.mp4", "*.avi", "*.mkv"])])
    file_name = os.path.basename(file_path)
    extract_pattern = lambda file_name: [part for part in file_name.split('_') if part.isdigit() and len(part) == 5][0] if "_" in file_name else None
    print("\nKITTI Dataset File Name:", file_name)
    input_dir = "Input"
    output_dir = "Output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    if file_path.endswith(('.png', '.jpg')):
        media_file = os.path.join(input_dir, "Input_Img.png")
        img = Image.open(file_path)
        img.save(media_file)
        img.show()
    elif file_path.endswith(('.mp4', '.avi', '.mkv')):
        media_file = os.path.join(input_dir, "Input_Video.mp4")
        shutil.copy(file_path, media_file)
        cap = cv2.VideoCapture(media_file)
        ret, frame = cap.read()
        image_path = os.path.join(input_dir, "Input_Img.png")
        cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        img = Image.open(image_path)
        img.show()
    print("\nKITTI Dataset loaded successfully...\n")
    print("\nNext, click the PREPROCESSING Button...\n")

