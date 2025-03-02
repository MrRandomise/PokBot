import random
import numpy as np
import cv2
import RecognitionText
import Test
import random
# Загрузка изображения стола
#image = cv2.imread(r"C:\Users\dmitr\Desktop\Poker\Data\screenshot_25.png")
image = cv2.imread(r"C:\Users\dmitr\Desktop\Poker\Data\screenshot_52.png")
height, width, _ = image.shape
#центр изоброжения
center_x = width // 2
center_y = height // 2
center = (center_x, center_y)
#размеры области игрока
rect_x = round(0.131 * width)
rect_y = round(0.21 * height)
rect_panel_y = round(0.044 * height)
#размеры области стола
axes_x = round(0.85 * center_x)
axes_y = round(0.68 * center_y)
axes = (axes_x, axes_y)
#размеры и координаты карт стола
cards_w = round(0.07 * width)
cards_h = round(0.13 * height)
cards_x = round(0.33 * width)
cards_y = center_y - round(0.12 * center_y)
cards_num = 5
offset = round(0.015 * width)

PlayaerCards_h = round(0.09 * height)
PlayaerCards_w = round(0.046 * width)

elipse_angle = 0
startAngle = 0
endAngle = 360
color = (0, 255, 0)
thickness = 3
num_players = 9

#print(height, width)
#cv2.ellipse(image, (center_x, center_y), axes, elipse_angle, startAngle, endAngle, color, thickness)

#Вычисляем позиции игроков
angles = np.radians(np.linspace(0, -360, num_players, endpoint=False))  # Начало с угла pi (снизу эллипса)
#angles = np.radians([270, 227, 175, 136.8, 104, 76, 45, 5, 314])
#player_offset = [-1, round(0.042 * height), round(0.017 * height), round(0.040 * height), round(0.0045 * center_y),round(0.002 * center_y), round(0.033 * height), round(0.021 * height), round(0.042 * height)]
#angles = np.radians([270, 205, 130, 50, 335])
Test.testDraw(image, center, axes, 10)
# Рисуем позиции игроков
for i, angle in enumerate(angles):
    # Вычисляем позицию каждого игрока
    x = int(center_x + axes_x * np.cos(angle))
    y = int(center_y - axes_y * np.sin(angle))  # Минус для инверсии оси Y, чтобы угол двигался против часовой стрелки
    # Создаем прямоугольник
    top_left = (x - rect_x // 2, y - rect_y // 2)
    bottom_right = (x + rect_x // 2, y + rect_y // 2)
    cv2.rectangle(image, top_left, bottom_right, color, thickness)

    #поле с блайндами
    blinds_left = (x - rect_x // 2, bottom_right[1] - rect_panel_y)
    blinds_right = (x + rect_x // 2, bottom_right[1])
    
    #print(RecognitionText.Recogn(image, blinds_left, blinds_right, r'--oem 3 --psm 11 outputbase digits', random.randint(0, 100000)))
    #cv2.rectangle(image, blinds_left, blinds_right, (255, 0, 0), thickness)
    #поле с именем
    name_left = (x - rect_x // 2, blinds_left[1] - rect_panel_y + 9)
    name_right = (x + rect_x // 2, blinds_left[1] + 2)
    #print(RecognitionText.Recogn(image, name_left, name_right, r'--oem 3 --psm 11', random.randint(0, 100000)))
    #cv2.rectangle(image, name_left, name_right, (0, 0, 255), thickness)
    #Распознаем карты игрока
    if i == 0:
        cards_one_left = (top_left[0] , top_left[1] )
        cards_one_right = (top_left[0] + PlayaerCards_w, top_left[1] + PlayaerCards_h )
        #cv2.rectangle(image, cards_one_left, cards_one_right, (0, 0, 255), thickness)
        cards_one_left = (top_left[0] +  PlayaerCards_w + 5, top_left[1] )
        cards_one_right = (top_left[0] + PlayaerCards_w * 2, top_left[1] + PlayaerCards_h )
        #cv2.rectangle(image, cards_one_left, cards_one_right, (0, 0, 255), thickness)
    # Рисуем точку для игрока
    #cv2.circle(image, (x, y), 10, (0, 0, 255), -1)

offset_x = 0
#Рисуем координат карт на флопе
for i in range(cards_num):
    x = cards_x + offset_x
    cards_top_left = (x - cards_w // 2, cards_y - cards_h // 2 + 20)
    cards_bottom_right= (x + cards_w // 2, cards_y + cards_h // 2 + 20)    
    offset_x += cards_w + offset;
    cv2.rectangle(image, cards_top_left, cards_bottom_right, color, thickness)
cv2.imshow("Player Positions", image)
cv2.waitKey(0)