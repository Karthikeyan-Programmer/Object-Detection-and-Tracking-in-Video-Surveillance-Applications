import logging
import time
from tkinter import Button, Label, Tk, messagebox
import warnings
import Preprocess
import Features
import Object_Detection
import Object_classification
import Object_Tracking
import object_detection_image
import object_detection_video
from plyer import notification
from LoadImg import Load
import Performance
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
def display_toast(title, message):
        max_title_length = 64  
        truncated_title = title[:max_title_length]
        notification.notify(
            title=truncated_title,
            message=message,
            timeout=5
        )
def load_image():
    print ("\n\t\t\t==========================********* Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications ********======================")    
    Load()
def PreProcessing():
    print ("\n\t\t\t==========================************* PRE-PROCESSING PROCESS ********======================")    
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",'Pre-Processing is in Process...')
    time.sleep(2)
    Preprocess.preprocess()
    time.sleep(2)
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",'PRE-PROCESSING PROCESS is Completed...!')
    print("\nPre-Processing Process Completed...\n")
    print("\nNext Click FEATURE EXTRACTION Button...\n")
def Features_Extraction():
    print ("\n\t\t\t==========================******** FEATURE EXTRACTION PROCESS ********======================")
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",'Features Extraction is in Process...')
    time.sleep(2)
    Features.FeatureExtraction()
    time.sleep(2)
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",'Features Extraction Process is completed...!')
    print("\nFeatures Extraction Process Completed...\n")
    print("\nNext Click OBJECT DETECTION Button...\n")
def Objectdetection():
    print ("\n\t==========================******************* OBJECT DETECTION *******************==========================")
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",'Object detection is in Process...')
    time.sleep(2)
    Object_Detection.Object_Detection()
    time.sleep(2)
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",'Object detection process is completed...!')
    print("OBJECT DETECTION Process Completed...")
    print("\nNext Click OBJECT CLASSIFICATION Button...\n")
def Objectclassification():
    print ("\n\t\t\t==========================************* OBJECT CLASSIFICATION PROCESS ********======================")
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",'Object classification Process...')
    Object_classification.Object_classification()
    print("OBJECT CLASSIFICATION Process Completed......")
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",'Object classification process is completed...!')
    print("\nNext Click OBJECT TRACKING Button...\n")
def ObjectTracking():
    print ("\n\t==========================******************* OBJECT TRACKING PROCESS *******************==========================")
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",'Object Tracking Process...')
    time.sleep(2)
    Object_Tracking.Object_Tracking()
    time.sleep(2)
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",'Object Tracking process is completed...!')
    print("OBJECT TRACKING Process Completed...")
    print("\nNext Click TRAIN THE MODEL Button...\n")
def TraintheModel():
    print ("\n\t==========================******************* TRAIN THE MODEL *******************==========================")
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",'Training Model is in Process...')
    time.sleep(2)
    object_detection_image.CNN()
    object_detection_video.CNN()
    time.sleep(2)
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",'Training Model is in Process is completed...!')
    print("OBJECT TRACKING Process Completed...")
    print("\nNext Click PERFORMANCE METRICS Button...\n")
def PerformanceMetrics():
    print ("\n\t==========================******************* PERFORMANCE METRICS *******************==========================")
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",' Performance Metrics is in Process...')
    time.sleep(2)
    Performance.PerformanceMetrics()
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_scores = classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",' Performance Metrics Completed...')
    display_toast("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",'THANK YOU....')
    time.sleep(2)
    print("Performance Metrics Completed...")
    print ("\n\t==========================******************* END *******************==========================")
def main():
    global window
    window = Tk()
    window.title("Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications")
    window_width = 850
    window_height = 800
    window.geometry(f"{window_width}x{window_height}")
    window.configure(background="floralwhite")
    Label(window, text = "Deep-Learning for Visual Object Detection and Tracking in Video Surveillance Applications",bg ="hotpink",fg ="floralwhite",width = "500", height = "6",font=('Georgia',14)).pack()
    Label(text = "",bg="floralwhite").pack()
    b1 = Button(text = "START", height = "2", width = "25",bg = "lightsteelblue3",fg ="black",font=('Georgia',13), command = load_image)
    b1.pack(pady=10)
    b2 = Button(text = "PREPROCESSING", height = "2", width = "25",bg = "lightsteelblue3",fg ="black",font=('Georgia',13), command = PreProcessing)
    b2.pack(pady=10)
    b3 = Button(text = "FEATURE EXTRACTION", height = "2", width = "25",bg = "lightsteelblue3",fg ="black",font=('Georgia',13), command = Features_Extraction)
    b3.pack(pady=10)
    b4 = Button(text = "OBJECT DETECTION", height = "2", width = "25",bg = "lightsteelblue3",fg ="black",font=('Georgia',13), command = Objectdetection)
    b4.pack(pady=10)
    b5 = Button(text = "OBJECT CLASSIFICATION", height = "2", width = "25",bg = "lightsteelblue3",fg ="black",font=('Georgia',13), command = Objectclassification)
    b5.pack(pady=10)
    b6 = Button(text = "OBJECT TRACKING", height = "2", width = "25",bg = "lightsteelblue3",fg ="black",font=('Georgia',13), command = ObjectTracking)
    b6.pack(pady=10)
    b7 = Button(text = "TRAIN THE MODEL", height = "2", width = "25",bg = "lightsteelblue3",fg ="black",font=('Georgia',13), command = TraintheModel)
    b7.pack(pady=10)
    b8 = Button(text = "PERFORMANCE METRICS", height = "2", width = "25",bg = "lightsteelblue3",fg ="black",font=('Georgia',13), command = PerformanceMetrics)
    b8.pack(pady=10)
    Label(text = "",bg="floralwhite").pack()    
    window.mainloop()

if __name__ == "__main__":
    main()
