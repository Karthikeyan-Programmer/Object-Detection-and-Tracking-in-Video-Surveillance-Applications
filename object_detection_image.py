import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
def CNN():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    img_path = 'Input\\Input_Img.png'  
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    predictions = model.predict(img_array)
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_array)
    for layer, activation in zip(model.layers, activations):
        print(f"Layer: {layer.name}, Output Shape: {activation.shape}")
    image = cv2.imread('Input\\Input_Img.png')
    if image is None:
        print("Error: Image not loaded.")
        exit()
    print("Original Image Shape:", image.shape)
    image = cv2.resize(image, (640, 480))
    print("Resized Image Shape:", image.shape)
    weights = "ssd_mobilenet/frozen_inference_graph.pb"
    model = "ssd_mobilenet/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(weights, model)
    class_names = []
    with open("ssd_mobilenet/coco_names.txt", "r") as f:
        class_names = f.read().strip().split("\n")
    blob = cv2.dnn.blobFromImage(
        image, 1.0/127.5, (320, 320), [127.5, 127.5, 127.5])
    net.setInput(blob)
    output = net.forward()
    for detection in output[0, 0, :, :]:
        probability = detection[2]
        if probability < 0.5:
            continue
        box = [int(a * b) for a, b in zip(detection[3:7], [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])]
        box = tuple(box)
        cv2.rectangle(image, box[:2], box[2:], (0, 255, 0), thickness=2)
        class_id = int(detection[1])
        label = f"{class_names[class_id - 1].upper()} {probability * 100:.2f}%"
        cv2.putText(image, label, (box[0], box[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

