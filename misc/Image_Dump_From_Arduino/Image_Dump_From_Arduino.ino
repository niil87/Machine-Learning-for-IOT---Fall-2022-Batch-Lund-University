#include <Arduino_OV767X.h>

#define USE_LOW_RES 1

#if USE_LOW_RES
  byte pixels[176 * 144 * 2];  //QCIF: 176x144 X 2 bytes per pixel (RGB565)
#else
  byte pixels[320 * 240 * 2]; //QVGA
#endif


void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("OV767X Camera Capture");
  Serial.println();

#if USE_LOW_RES
  if (!Camera.begin(QCIF, RGB565, 1)) {
#else
  if (!Camera.begin(QVGA, RGB565, 1)) {
#endif
    Serial.println("Failed to initialize camera!");
    while (1);
  }

  Serial.println("Camera settings:");
  Serial.print("\twidth = ");
  Serial.println(Camera.width());
  Serial.print("\theight = ");
  Serial.println(Camera.height());
  Serial.print("\tbits per pixel = ");
  Serial.println(Camera.bitsPerPixel());
  Serial.println();
}

void takePicture() {
  Camera.readFrame(pixels);
  Serial.println("-------- BEGIN IMAGE --------");
  int bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
  Serial.write(pixels, bytesPerFrame);
}

void loop() {
  if (Serial.read() == 'c') {
    takePicture();
  }
}
