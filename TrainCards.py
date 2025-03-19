import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

 # s - пики, d - бубны, c - трефы, h - червы
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

    s = x+y
    return int(s)

def StartTrain():

    # Загрузка обучающего набора данных
    train_data = pd.read_csv(r"C:\Users\Dmitry\Desktop\DataSet\Cards\Valid\train_labels.csv")
    train_images = []
    train_labels = []
    for index, row in train_data.iterrows():
        fileTrain = "C:\\Users\Dmitry\\Desktop\\DataSet\\Cards\\Valid\\train\\train\\" + row['filename']
        img = load_img(fileTrain, target_size=(970, 700))  # Укажите ширину и высоту
        img_array = img_to_array(img)
        train_images.append(img_array)
        train_class = safe_convert_to_int(row['class'])
        train_labels.append(train_class)

    # Загрузка тестового набора данных
    test_data = pd.read_csv(r"C:\Users\Dmitry\Desktop\DataSet\Cards\Valid\test_labels.csv")
    test_images = []
    test_labels = []
    for index, row in test_data.iterrows():
        fileTest = "C:\\Users\Dmitry\\Desktop\\DataSet\\Cards\\Valid\\test\\test\\" + row['filename']
        img = load_img(fileTest, target_size=(970, 700))  # Укажите ширину и высоту
        img_array = img_to_array(img)
        test_images.append(img_array)
        test_class = safe_convert_to_int(row['class'])
        test_labels.append(test_class)

    # Преобразование списков в массивы NumPy
    X_train = np.array(train_images)
    y_train = np.array(train_labels)
    X_test = np.array(test_images)
    y_test = np.array(test_labels)

    # Нормализация значений пикселей изображений
    X_test = X_test.astype('float32') / 255.0


    # Преобразование меток в категориальный формат
    num_classes = 94  # Количество уникальных классов
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    # Создание модели
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(970, 700, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Компиляция модели
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Обучение модели
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    # Сохранение модели
    model.save('playing_card_classifier.h5')

    print("Finish! Model save 'playing_card_classifier.h5'")