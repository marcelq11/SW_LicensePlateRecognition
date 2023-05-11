import cv2
import numpy as np
import math
import os

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def warp_img(contour,img):
    pts1 = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])
    pts2 = np.float32([[0, 0], [1040, 0], [0, 224], [1040, 224]])
    cont_sorted = sorted(contour, key=lambda x: x[0][0])
    cont_sorted_left = sorted(cont_sorted[:2], key=lambda x: x[0][1])
    cont_sorted_right = sorted(cont_sorted[2:], key=lambda x: x[0][1])
    pts1 = np.float32([cont_sorted_left[0][0], cont_sorted_right[0][0], cont_sorted_left[1][0], cont_sorted_right[1][0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (1040, 224))
    return warped



def find_contours(img):
    temp = 0
    temp_list = []
    warped = np.zeros((224, 1040, 3), dtype = np.uint8)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)# cv2.FILLED

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.015 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.contourArea(contour) > 20000:
            max_len = -1
            for i in range(4):
                side_len = cv2.norm(approx[i % 4] - approx[(i + 1) % 4])
                if side_len > max_len:
                    max_len = side_len
            min_len = max_len
            for i in range(4):
                side_len = cv2.norm(approx[i % 4] - approx[(i + 1) % 4])
                if side_len < min_len:
                    min_len = side_len
            ratio = max_len / min_len
            if ratio > 3 and ratio < 6 and max_len < int(0.8 * img.shape[1]):
                temp_list.append(approx)
                # cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
                temp += 1
    if temp > 0:
        temp_size = 99999999
        for i in temp_list:
            if cv2.contourArea(i) < temp_size:
                temp_size = cv2.contourArea(i)
                temp_contour = i
        for i in range(temp_contour.shape[0]):
            warped = warp_img(temp_contour, img)
    return warped, temp

def find_plate(img):
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    warped, temp = find_contours(img_blur)
    # cv2.drawContours(img, [temp_contour], 0, (0, 0, 255), 2)
    if temp == 0:
        warped = find_plate2(img)
    return warped

def find_plate2(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.bilateralFilter(img_gray, 15, 35, 100)
    warped, temp = find_contours(img_blur)
    return warped


for img in load_images_from_folder('train'):
    img = find_plate(img)
    while(1):
        cv2.imshow('img', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()