#!/usr/bin/python3
import os
from PIL import Image
import numpy as np

def convert(data: np.ndarray) -> Image:
    width, height = data.shape
    img = Image.new('RGB', (width, height))

    for x in range(width):
        for y in range(height):
            pixel = data[x, y]
            r = (pixel & 0xF800) >> 11 # F800: bits 11-15
            g = (pixel & 0x07E0) >> 5  # 07E0: bits 5-10
            b = (pixel & 0x1F)         # 001F: bits 0-4
            img.putpixel((x, y), (r * 8, g * 4, b * 8)) # multiply by (8, 4, 8) to scale back to [0,255]

    return img

f = 'celeb_transformed/Angelina Jolie/001_fe3347c0.npy'
data = np.load(f)
img = convert(data)
print(img.mode, img.size)
print(img.getpixel((50, 50)))
img.save('test.png', 'PNG')