import cv2
import numpy as np

def testDraw(image, center, axes, step):

    for angle in np.arange(0, 361, step):  # От 0 до 360 с шагом 10
        radians = np.radians(angle)  # Преобразуем угол в радианы
        x = int(center[0] + axes[0] * np.cos(radians))  # X координата
        y = int(center[1] - axes[1] * np.sin(radians))  # Y координата
    
    # Рисуем линию от центра до окружности
        cv2.line(image, center, (x, y), (0, 0, 255), 1)

    # Добавляем текст с углом
        text_position = (int(center[0] + (axes[0] + 20) * np.cos(radians)), 
                     int(center[1] - (axes[1] + 20) * np.sin(radians)))
        cv2.putText(image, str(angle), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def TableDraw():

    # Параметры изображения и игроков
    image_width = 800
    image_height = 600
    image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255  # Создаем белое изображение

    # Центр овального стола и его размеры
    center_x = image_width // 2
    center_y = image_height // 2
    axes_x = 200  # радиус по оси x
    axes_y = 100  # радиус по оси y
    num_players = 9
    rect_x = 40  # ширина прямоугольника
    rect_y = 20  # высота прямоугольника
    color = (0, 0, 255)  # Цвет (красный)
    thickness = -1  # Заполненный прямоугольник

    # Вычисляем позиции игроков
    angles =  np.linspace(0, 2 * np.pi, num_players, endpoint=False)
    # Рисуем позиции игроков
    for i, angle in enumerate(angles):
        # Вычисляем позицию каждого игрока
        x = int(center_x + axes_x * np.cos(angle))
        y = int(center_y + axes_y * np.sin(angle))  # отрицательный знак убираем для правильного направления
        # Создаем прямоугольник
        top_left = (x - rect_x // 2, y - rect_y // 2)
        bottom_right = (x + rect_x // 2, y + rect_y // 2)
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

    # Сохраняем или отображаем изображение
    cv2.imshow('Players on Oval Table', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()