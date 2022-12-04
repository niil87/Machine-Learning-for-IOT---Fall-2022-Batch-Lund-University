#!/usr/bin/python3
import os
from PIL import Image

for celeb in os.listdir('celeb_original'):
    print(f'------- {celeb} -------')
    path = os.path.join('celeb_original', celeb)

    for img in os.listdir(path):
        file = os.path.join(path, img)
        im = Image.open(file)
        print(img, ': ', im.format, im.size)