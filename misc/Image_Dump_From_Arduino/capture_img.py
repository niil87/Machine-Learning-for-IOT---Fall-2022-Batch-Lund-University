#!/usr/bin/python3
import argparse
import serial
import numpy as np
from PIL import Image


# returns the read image as a 2-dimensional array of pixel values
def read_img(ser, width, height, bytes_per_pixel=2, verbose=False):
    print('waiting for image header...')

    line = ser.readline()
    while line != b'-------- BEGIN IMAGE --------\r\n':
        print(line)
        line = ser.readline()

    img = []
    for i in range(height):
        row = []
        for j in range(width):
            input = ser.read(bytes_per_pixel) # multiply by 2 because each byte is represented by 2 hex characters
            pixel = int.from_bytes(input, 'big')
            
            if verbose:
                print(f'pixel {i}, {j}: {input} ({pixel})')

            row.append(pixel)
        img.append(row)
    return img

def save_numpy(img_data, filename):
    img_array = np.array(img_data, dtype=np.uint16)
    print(f'saving image as numpy array of shape {img_array.shape} to', filename)
    np.save(filename, img_array)

def pixel_from_565(pixel):
    r = (pixel & 0xF800) >> 11
    g = (pixel & 0x07E0) >> 5
    b = (pixel & 0x001F)

    return (r << 3, g << 2, b << 3)

def img_from_565(img_data):
    img = Image.new('RGB', (len(img_data[0]), len(img_data)))
    for i in range(len(img_data)):
        for j in range(len(img_data[0])):
            img.putpixel((j, i), pixel_from_565(img_data[i][j]))
    return img

def save_img(img_data, filename, color_format='RGB565'):
    img = None
    if color_format == 'RGB565':
        img = img_from_565(img_data)
    
    if img is None:
        print('ERROR: invalid input format: ', color_format)
        return
    
    print(f'saving image to', filename)
    img.save(filename)

def main():
    parser = argparse.ArgumentParser(description='read input from serial port')
    parser.add_argument('-p', '--port', required=True, type=str)
    parser.add_argument('-b', '--baud', type=int, default=9600)
    parser.add_argument('-v', '--verbose', action='store_true', help='print verbose output', default=False)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--bytes_per_pixel', type=int, default=2)
    parser.add_argument('--color_format', type=str, default='RGB565')
    parser.add_argument('--npy-output', metavar='FILE', type=str, help='output image in native color as numpy array to FILE')
    parser.add_argument('--img-output', metavar='FILE', type=str, help='output image in RGB888 form to FILE')

    args = parser.parse_args()
    
    port = args.port
    baud = args.baud
    verbose = args.verbose
    width = args.width
    height = args.height
    bytes_per_pixel = args.bytes_per_pixel
    npy_output = args.npy_output
    img_output = args.img_output
    
    if npy_output is None and img_output is None:
        parser.error('at least one of --npy-output or --img-output must be specified')

    with serial.Serial(port, baud) as ser:
        ser.write(b'c') # send 'c' to start capture
        img_data = read_img(ser, width, height, bytes_per_pixel, verbose=verbose)

        if npy_output is not None:
            save_numpy(img_data, npy_output)
        
        if img_output is not None:
            save_img(img_data, img_output)


if __name__ == '__main__':
    main()