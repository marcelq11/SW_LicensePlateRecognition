import cv2
import numpy as np
import json
import os
import sys


def load_images_from_folder(folder_path):
    keys = []
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            keys.append(filename)
            images.append(img)
    return keys, images


def warp_img(contour, img):
    pts1 = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])
    pts2 = np.float32([[0, 0], [1040, 0], [0, 224], [1040, 224]])
    cont_sorted = sorted(contour, key=lambda x: x[0][0])
    cont_sorted_left = sorted(cont_sorted[:2], key=lambda x: x[0][1])
    cont_sorted_right = sorted(cont_sorted[2:], key=lambda x: x[0][1])
    pts1 = np.float32(
        [cont_sorted_left[0][0], cont_sorted_right[0][0], cont_sorted_left[1][0], cont_sorted_right[1][0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (1040, 224))
    return warped


def find_contours(img):
    temp = 0
    temp_list = []
    warped = np.zeros((224, 1040, 3), dtype=np.uint8)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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


def find_plate2(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    warped, temp = find_contours(img_blur)
    return warped


def find_plate(img):
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.bilateralFilter(img_gray, 15, 35, 100)
    warped, temp = find_contours(img_blur)
    if temp == 0:
        warped = find_plate2(img)
    return warped


def filter_contours(contours):
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        is_contained = False
        for existing_contour in filtered_contours:
            existing_x, existing_y, existing_w, existing_h = cv2.boundingRect(existing_contour)
            if x >= existing_x and y >= existing_y and x + w <= existing_x + existing_w and y + h <= existing_y + existing_h:
                is_contained = True
                break
        if not is_contained:
            filtered_contours.append(contour)
    return filtered_contours

def compare_images(img1, img2):
    diff = cv2.absdiff(img1, img2)
    diff_sum = np.sum(diff)
    return diff_sum

def to_string(char_list):
    string = ""
    for char in char_list:
        string += char
    return string

dict = {}
keys, images = load_images_from_folder('font/pliki/')
results = {}

for key, img in zip(keys, images):
    dict[key[0]] = img
for key, img in dict.items():
    chosen_contours_characters = []
    cropped_images_character = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 60, 100)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        [X, Y, W, H] = cv2.boundingRect(contour)
        if H > 0.6 * img.shape[0] and H < 0.97 * img.shape[0]:
            chosen_contours_characters.append(contour)
    contour = filter_contours(chosen_contours_characters)
    [X, Y, W, H] = cv2.boundingRect(contour[0])
    cv2.rectangle(img, (X, Y), (X + W, Y + H), (255), 2)
    cropped_images_character = img[Y:Y + H, X:X + W]
    dict[key] = cropped_images_character
dict_images = {}
file_train_path = sys.argv[1]
keys, images = load_images_from_folder(file_train_path)

for filename, img in zip(keys, images):
    dict_images[filename] = img
for filename, img in dict_images.items():
    char_founded = []
    i = 0
    img = find_plate(img)
    chosen_contours = []
    cropped_images = []
    edges = cv2.Canny(img, 60, 100)
    ret, thresh1 = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        [X, Y, W, H] = cv2.boundingRect(contour)
        if H > 0.4 * img.shape[0] and H < 0.95 * img.shape[0] and W < 0.15 * img.shape[1] and W * H > 6000:
            chosen_contours.append(contour)
    contours = filter_contours(chosen_contours)
    contours = sorted(contours, key=lambda box: cv2.boundingRect(box)[0])
    for contour in contours:
        [X, Y, W, H] = cv2.boundingRect(contour)
        cv2.rectangle(img, (X, Y), (X + W, Y + H), (255), 2)
        cropped_images.append(thresh1[Y:Y + H, X:X + W])
    for cropped_image in cropped_images:
        cropped_image_blur = cv2.GaussianBlur(cropped_image, (3, 3), 0)
        ret, thresh = cv2.threshold(cropped_image_blur, 95, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cropped_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        lowest = 10000000
        char = ''
        for key, img_d in dict.items():
            img_d = cv2.resize(img_d, (cropped_image.shape[1], cropped_image.shape[0]))
            if compare_images(img_d, cropped_image) < lowest:
                lowest = compare_images(img_d, cropped_image)
                char = key
        char_founded.append(char)
    results[filename] = to_string(char_founded)

file_json_path = sys.argv[2] + '/results.json'
with open(file_json_path, 'w') as file:
    json.dump(results, file, indent=4)
