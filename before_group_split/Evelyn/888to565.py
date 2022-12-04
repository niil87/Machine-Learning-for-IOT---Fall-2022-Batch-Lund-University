#!/usr/bin/python3
import os
from PIL import Image
import numpy as np

def transform(im: Image) -> np.ndarray:
    width, height = im.size
    new_pxs = np.zeros(shape=(width, height), dtype=np.uint16)

    for x in range(width):
        for y in range(height):
            r, g, b = im.getpixel((x,y))

            # downsample to RGB565
            r //= 8
            g //= 4
            b //= 8
            new_pxs[x, y] = (r << 11) | (g << 5) | b

    return new_pxs

for celeb in os.listdir('celeb_cropped'):
    print(f'------- {celeb} -------')
    path = os.path.join('celeb_cropped', celeb)
    new_path = os.path.join('celeb_transformed', celeb)

    os.makedirs(new_path, exist_ok=True)

    for img_file in os.listdir(path):
        print(img_file)

        file = os.path.join(path, img_file)
        im = Image.open(file)
        out = transform(im)

        img_file_noext, _ = os.path.splitext(img_file)
        new_file = os.path.join(new_path, img_file_noext + '.npy')
        np.save(new_file, out)
