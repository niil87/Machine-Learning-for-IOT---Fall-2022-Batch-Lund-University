#!/usr/bin/python3
import os
from PIL import Image

for celeb in os.listdir('celeb_original'):
    print(f'------- {celeb} -------')
    path = os.path.join('celeb_original', celeb)
    new_path = os.path.join('celeb_scaled', celeb)

    os.makedirs(new_path, exist_ok=True)

    for img_file in os.listdir(path):
        print(img_file)

        file = os.path.join(path, img_file)
        new_file = os.path.join(new_path, img_file)
        im = Image.open(file)

        w, h = im.size
        if w < 320 or h < 240:
            continue
        
        new_w = 320
        new_h = round(h * (320 / w))

        out = im.resize((new_w, new_h))
        out.save(new_file)