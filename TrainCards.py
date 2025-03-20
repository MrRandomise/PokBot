import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image #  pip install Pillow
from sklearn.preprocessing import LabelEncoder

# s - ����, d - �����, c - �����, h - �����

# s - 0, d - 1, c - 2, h - 3
def safe_convert_to_int(data):
    x = data[0]
    y = data[1]
    if (x == "t"):
        x = "-1"
    if (x == "j"):
        x = "-2"
    if (x == "q"):
        x = "-3"
    if (x == "k"):
        x = "-5"
    if (x == "a"):
        x = "0"
    if (y == "s"):
        y = "0"
    if (y == "d"):
        y = "1"
    if (y == "c"):
        y = "2"
    if (y == "h"):
        y = "3"

    s = x + y
    return int(s)

def build_model(input_shape, num_classes):
    # ������� ����
    input_layer = Input(shape=input_shape)

    # ���������� ����
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    # ����� ��� �������������
    classification_output = Dense(num_classes, activation='softmax', name='classification_output')(x)

    # ����� ��� ��������� (���������� bounding box)
    regression_output = Dense(4, activation='linear', name='regression_output')(x)

    # �������� ������
    model = Model(inputs=input_layer, outputs=[classification_output, regression_output])
    return model

def StartTrain():
    # �������� ����������� GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # ��������� ��� ������������� ������ �� ���� �������������
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("GPU is available and memory growth is enabled.")

        except RuntimeError as e:
            print(e)
    else:
        print("GPU is not available. Training will run on CPU.")


    # �������� ���������� ������ ������
    #train_data = pd.read_csv(r"C:\Users\dmitr\Desktop\DataSet\Cards\Valid\train_labels.csv")
    train_data = pd.read_csv(r"C:\Users\Dmitry\Desktop\DataSet\Cards\Valid\train_labels.csv")
    train_images = []
    train_labels = []
    train_bboxes = []  # ������ ��� �������� ��������� bounding boxes

    for index, row in train_data.iterrows():
        fileTrain = "C:\\Users\\Dmitry\\Desktop\\DataSet\\Cards\Valid\\train\\train\\" + row['filename']
        img = load_img(fileTrain, target_size=(970, 700)) # ������� ������ � ������
        img_array = img_to_array(img)
        train_images.append(img_array)
        train_class = safe_convert_to_int(row['class'])
        #train_label = {'class': train_class, 'bbox': train_rec}
        train_labels.append(train_class)
        train_bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        train_bboxes.append(train_bbox)

    # �������� ��������� ������ ������
    #test_data = pd.read_csv(r"C:\Users\dmitr\Desktop\DataSet\Cards\Valid\test_labels.csv")
    test_data = pd.read_csv(r"C:\Users\Dmitry\Desktop\DataSet\Cards\Valid\test_labels.csv")
    test_images = []
    test_labels = []
    test_bboxes = []  # ������ ��� �������� ��������� bounding boxes

    for index, row in test_data.iterrows():
        #fileTest = "C:\\Users\dmitr\\Desktop\\DataSet\\Cards\\Valid\\test\\test\\" + row['filename']
        fileTest = "C:\\Users\\Dmitry\\Desktop\\DataSet\\Cards\\Valid\\test\\test\\" + row['filename']
        img = load_img(fileTest, target_size=(970, 700)) # ������� ������ � ������
        img_array = img_to_array(img)
        test_rec = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        test_images.append(img_array)
        test_class = safe_convert_to_int(row['class'])
        #test_label = {'class': test_class, 'bbox': test_rec}
        test_labels.append(test_class)
        # ��������� ����� ������
        test_class = safe_convert_to_int(row['class'])
        test_labels.append(test_class)

    # �������������� ������� � ������� NumPy
    X_train = np.array(train_images)
    y_train = np.array(train_labels)
    train_bboxes = np.array(train_bboxes)  # ���������� bounding boxes ��� ���������� ������

    X_test = np.array(test_images)
    y_test = np.array(test_labels)
    test_bboxes = np.array(test_bboxes)  # ���������� bounding boxes ��� ��������� ������


    # ������������ �������� �������� �����������
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # �������������� ����� � �������������� ������
    num_classes = 94  # ���������� ���������� �������
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    input_shape = (970, 700, 3)
    model = build_model(input_shape, num_classes)

    # ���������� ������
    with tf.device('/GPU:0'):
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'classification_output': 'categorical_crossentropy',  # ������ ��� �������������
                'regression_output': 'mean_squared_error'           # ������ ��� ���������
            },
            metrics={
                'classification_output': 'accuracy',                # ������� ��� �������������
                'regression_output': 'mae'                          # ������� ��� ���������
            }
        )
        # �������� ������
        history = model.fit(
            X_train,
            {
                'classification_output': y_train,  # ������� ����� ��� �������������
                'regression_output': train_bboxes  # ������� ���������� ��� ���������
            },
            validation_data=(
                X_test,
                {
                    'classification_output': y_test,
                    'regression_output': test_bboxes
                }
            ),
            epochs=10,
            batch_size=32
        )

    # ���������� ������
    model.save('playing_card_detector.h5')
    print("Finish! Model saved as 'playing_card_detector.h5'")