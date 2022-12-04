To transform the dataset (celeb_original) into cropped 320x240 RGB565 images, do:
```
./scale.py
./face_crop.py
./888to565.py
```
The RGB565 files are stored in celeb_transformed, in the form of pickled NumPy arrays of shape 320x240. The array can be read from a file with `data = np.load(filename)`. Each element is a uint16 containing the two bytes of data for each pixel. To extract the RGB components from an integer like this, you can do:

```
pixel = data[x, y]
r = (pixel & 0xF800) >> 11 # F800: bits 11-15
g = (pixel & 0x07E0) >> 5  # 07E0: bits 5-10
b = (pixel & 0x001F)       # 001F: bits 0-4
```
