import cv2
import freetype
import numpy as np

# Inicjalizacja obiektu FreeType
font_path = 'font1.ttf'
font_size = 74
font = freetype.Face(font_path)
font.set_char_size(font_size * 64)

# Ustalenie rozmiaru obrazka
image_width = 60
image_height = 60

# Tworzenie obrazków dla znaków alfabetu i liczb
characters = 'ABCDEFGHIJKLMNOPRSTUVWXYZ0123456789'
output_folder = '/home/marcel/SW_LicensePlateRecognition/testowy/pliki/'

for char in characters:
    # Ustalanie wymiarów obrazka
    font.load_char(char)
    bitmap = font.glyph.bitmap
    width, height = bitmap.width, bitmap.rows

    # Tworzenie obrazka
    image = np.ones((image_height, image_width), dtype=np.uint8) * 255
    left = (image_width - width) // 2
    top = (image_height - height) // 2
    for x in range(width):
        for y in range(height):
            image[top + y, left + x] = 255 - bitmap.buffer[y * width + x]

    # Zapisywanie obrazka
    cv2.imwrite(output_folder + char + '.png', image)

print('Zakończono generowanie obrazków znaków.')