import random
import numpy as np
from RJson import ReadJson
from RJson import GetRectangle
import cv2
from RecognitionText import Recogn
import Test
import random
from TrainCards import StartTrain
# Загрузка изображения стола
#image = cv2.imread(r"C:\Users\dmitr\Desktop\Poker\Data\screenshot_25.png")
file = r"C:\Users\Dmitry\Desktop\DataSet\Table\Old\5\screenshot_91.png"
image = cv2.imread(file)
height, width, _ = image.shape
#центр изоброжения
center_x = width // 2
center_y = height // 2
center = (center_x, center_y)

#размеры и координаты карт стола
cards_w = round(0.07 * width)
cards_h = round(0.13 * height)
cards_x = round(0.33 * width)
cards_y = center_y - round(0.12 * center_y)
cards_num = 5
offset = round(0.015 * width)

PlayaerCards_h = round(0.09 * height)
PlayaerCards_w = round(0.046 * width)

color = (0, 255, 0)
thickness = 1
StartTrain()


#print(height, width)
#cv2.ellipse(image, (center_x, center_y), axes, elipse_angle, startAngle, endAngle, color, thickness)

#Вычисляем позиции игроков
#angles = np.radians(np.linspace(0, -360, num_players, endpoint=False))  # Начало с угла pi (снизу эллипса)
#angles = np.radians([270, 227, 175, 136.8, 104, 76, 45, 5, 314])
#player_offset = [-1, round(0.042 * height), round(0.017 * height), round(0.040 * height), round(0.0045 * center_y),round(0.002 * center_y), round(0.033 * height), round(0.021 * height), round(0.042 * height)]
#angles = np.radians([270, 205, 130, 50, 335])
#Test.testDraw(image, center, axes, 10)


temp = ReadJson(r"C:\Users\Dmitry\Desktop\DataSet\5Max.json") 
data = GetRectangle(temp)
# Рисуем позиции игроков
#for i, temp in enumerate(data):
    # Вычисляем позицию каждого игрока
    #x = int(center_x + axes_x * np.cos(angle))
    #y = int(center_y - axes_y * np.sin(angle))  # Минус для инверсии оси Y, чтобы угол двигался против часовой стрелки
    # Создаем прямоугольник


    #try:
        #position = data[i][4].index("Position")
    #except ValueError:
         #print(Recogn(image, data[i][0], data[i][1], data[i][2], data[i][3], r'--oem 3 --psm 11 outputbase'))
    #op_left = (data[i][0], data[i][1])
    #bottom_right = (data[i][0] + data[i][2], data[i][1] + data[i][3])
    #cv2.rectangle(image, top_left, bottom_right, color, thickness)

    #поле с блайндами
    #blinds_left = (x - rect_x // 2, bottom_right[1] - rect_panel_y)
    #blinds_right = (x + rect_x // 2, bottom_right[1])
    
    #print(RecognitionText.Recogn(image, blinds_left, blinds_right, r'--oem 3 --psm 11 outputbase digits', random.randint(0, 100000)))
    #cv2.rectangle(image, blinds_left, blinds_right, (255, 0, 0), thickness)
    #поле с именем
    #name_left = (x - rect_x // 2, blinds_left[1] - rect_panel_y + 9)
    #name_right = (x + rect_x // 2, blinds_left[1] + 2)
    #print(RecognitionText.Recogn(image, name_left, name_right, r'--oem 3 --psm 11', random.randint(0, 100000)))
    #cv2.rectangle(image, name_left, name_right, (0, 0, 255), thickness)
    #Распознаем карты игрока
    #if i == 0:
        #cards_one_left = (top_left[0] , top_left[1] )
        #cards_one_right = (top_left[0] + PlayaerCards_w, top_left[1] + PlayaerCards_h )
        #cv2.rectangle(image, cards_one_left, cards_one_right, (0, 0, 255), thickness)
        #cards_one_left = (top_left[0] +  PlayaerCards_w + 5, top_left[1] )
        #cards_one_right = (top_left[0] + PlayaerCards_w * 2, top_left[1] + PlayaerCards_h )
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