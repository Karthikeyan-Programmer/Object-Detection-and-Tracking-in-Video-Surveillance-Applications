import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import pickle
def Object_classification():
    pickle_file_path = 'Output/Test_Accuracy.pickle'
    with open(pickle_file_path, 'rb') as file:
        test_accuracy = pickle.load(file)
    def create_model(input_shape, num_classes):
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    train_data_dir = 'Dataset\\test'
    test_data_dir = 'Dataset\\train'
    img_size = (224, 224)
    num_classes = 10
    model = create_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 32
    print("Training Data:")
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    print("Number of classes:", len(train_generator.class_indices))
    print("Class labels:", train_generator.class_indices)

    print("\nTesting Data:")
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    print("Number of classes:", len(test_generator.class_indices))
    print("Class labels:", test_generator.class_indices)
    accuracy = test_accuracy
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    data_to_save = {
        'model': model,
        'train_generator': train_generator,
        'test_generator': test_generator,
        'img_size': img_size,
        'num_classes': num_classes
    }
