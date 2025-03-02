import pytesseract
import numpy as np
from PIL import Image
import cv2

def Recogn(image, left_top_rec, right_bottom_rec, custom_config, i):

    x, y = left_top_rec
    x2, y2 = right_bottom_rec
    w = x2 - x
    h = y2 - y
    roi = image[y:y+h, x:x+w]
 
    # Преобразование в оттенки серого и бинаризация
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #_, binary_roi = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)
    #roi_pil = Image.fromarray(binary_roi)
    #kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    #sharpened_image = cv2.filter2D(binary_image, -1, kernel)
    cv2.imshow("Player Positions", gray_roi)
    cv2.waitKey(0)

    if roi.shape[0] == 0 or roi.shape[1] == 0:
        print("Error: ROI null.")
    text = ""
    if roi.size == 0:
        print("ROI is empty!")
    try:
        cv2.imwrite(rf'C:\Users\dmitr\Desktop\Poker\Data\Test\screenshot_{i}.png', roi)
        #text = pytesseract.image_to_string(roi)
        text = pytesseract.image_to_string(gray_roi, config=custom_config, lang="eng")
    except Exception as e:
        print(f"Error while recognizing text: {e}")
 
    return text