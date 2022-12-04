#!/usr/bin/python3

import cv2
import os
from PIL import Image

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def largest_face(faces):
    return max(faces, key=lambda x: x[2] * x[3])

def get_crop_coords(path):
    # Read the input image
    img = cv2.imread(path)
    h_img, w_img, _ = img.shape
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None

    # Draw rectangle around the faces
    x, y, w, h = largest_face(faces)
    
    x_center = x + w//2
    y_center = y + h//2

    x1, y1, x2, y2 = x_center - 160, y_center - 120, x_center + 160, y_center + 120

    # make sure cropping zone stays within image
    if x1 < 0:
        x1 = 0
        x2 = 320
    elif x2 > w_img:
        x1 = w_img - 320
        x2 = w_img
    
    if y1 < 0:
        y1 = 0
        y2 = 240
    elif y2 > h_img:
        y1 = h_img - 240
        y2 = h_img
    
    assert x2 - x1 == 320
    assert y2 - y1 == 240

    return x1, y1, x2, y2

for celeb in os.listdir('celeb_scaled'):
    print(f'------- {celeb} -------')
    path = os.path.join('celeb_scaled', celeb)
    new_path = os.path.join('celeb_cropped', celeb)

    os.makedirs(new_path, exist_ok=True)

    for img_file in os.listdir(path):
        print(img_file)

        file = os.path.join(path, img_file)
        new_file = os.path.join(new_path, img_file)
        im = Image.open(file)

        w, h = im.size
        if w < 320 or h < 240:
            continue
        
        coords = get_crop_coords(file)
        if coords is None:
            continue
        x1, y1, x2, y2 = coords

        out = im.crop((x1, y1, x2, y2))

        out.save(new_file)
