import cv2
import numpy as np

def testDraw(image, center, axes, step):

    for angle in np.arange(0, 361, step):  # �� 0 �� 360 � ����� 10
        radians = np.radians(angle)  # ����������� ���� � �������
        x = int(center[0] + axes[0] * np.cos(radians))  # X ����������
        y = int(center[1] - axes[1] * np.sin(radians))  # Y ����������
    
    # ������ ����� �� ������ �� ����������
        cv2.line(image, center, (x, y), (0, 0, 255), 1)

    # ��������� ����� � �����
        text_position = (int(center[0] + (axes[0] + 20) * np.cos(radians)), 
                     int(center[1] - (axes[1] + 20) * np.sin(radians)))
        cv2.putText(image, str(angle), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def TableDraw():

    # ��������� ����������� � �������
    image_width = 800
    image_height = 600
    image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255  # ������� ����� �����������

    # ����� ��������� ����� � ��� �������
    center_x = image_width // 2
    center_y = image_height // 2
    axes_x = 200  # ������ �� ��� x
    axes_y = 100  # ������ �� ��� y
    num_players = 9
    rect_x = 40  # ������ ��������������
    rect_y = 20  # ������ ��������������
    color = (0, 0, 255)  # ���� (�������)
    thickness = -1  # ����������� �������������

    # ��������� ������� �������
    angles =  np.linspace(0, 2 * np.pi, num_players, endpoint=False)
    # ������ ������� �������
    for i, angle in enumerate(angles):
        # ��������� ������� ������� ������
        x = int(center_x + axes_x * np.cos(angle))
        y = int(center_y + axes_y * np.sin(angle))  # ������������� ���� ������� ��� ����������� �����������
        # ������� �������������
        top_left = (x - rect_x // 2, y - rect_y // 2)
        bottom_right = (x + rect_x // 2, y + rect_y // 2)
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

    # ��������� ��� ���������� �����������
    cv2.imshow('Players on Oval Table', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()