#include <ArduinoBLE.h>

typedef struct __attribute__( ( packed ) )
{
  uint8_t turn;
  int16_t w0;
  int16_t w1;
  int16_t w2;
} weights_data_t;

weights_data_t weightsData;

BLEService weightsService("19B10000-E8F2-537E-4F6C-D104768A1214");  // BLE Weights Service

BLECharacteristic weightsCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite | BLENotify, sizeof(weightsData));
const int ledPin = LED_BUILTIN;  // pin to use for the LED
unsigned long last_time = 0;

void ConnectHandler(BLEDevice central) {
  // central connected event handler
  Serial.print("Connected event, central: ");
  Serial.println(central.address());
  BLE.advertise();
}

void DisconnectHandler(BLEDevice central) {
  // central disconnected event handler
  Serial.print("Disconnected event, central: ");
  Serial.println(central.address());
  BLE.advertise();
}

void setup() {
  Serial.begin(9600);
  //while (!Serial);

  // set LED pin to output mode
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, HIGH);

  // begin initialization
  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");

    while (1)
      ;
  }
  BLE.setEventHandler(BLEConnected, ConnectHandler);
  BLE.setEventHandler(BLEDisconnected, DisconnectHandler);
  // set advertised local name and service UUID:
  BLE.setLocalName("MLMaster");
  BLE.setAdvertisedService(weightsService);

  // add the characteristic to the service
  weightsService.addCharacteristic(weightsCharacteristic);

  // add service
  BLE.addService(weightsService);
  // set the initial value for the characeristic:
  weightsData.turn = 1;
  weightsCharacteristic.writeValue((uint8_t *)&weightsData, sizeof(weightsData));

  // start advertising
  BLE.advertise();

  Serial.println("BLE Weights Peripheral");
}

void loop() {
  unsigned long current_time = millis();
  BLE.poll();
  
    if (weightsCharacteristic.written()) {
      // Below can be used to read new weights and applying them to current weights, for now we just copy
      weightsData = *((weights_data_t *)weightsCharacteristic.value());
      //weightsCharacteristic.readValue((uint8_t *)&weightsData, sizeof(weightsData)); // Also works, but copies directly
      Serial.print("Peripheral Weights Value: ");
      Serial.print(weightsData.turn);
      Serial.print(" ");
      Serial.print(weightsData.w0);
      Serial.print(" ");
      Serial.print(weightsData.w1);
      Serial.print(" ");
      Serial.println(weightsData.w2);
  
      if (weightsData.turn == 0) {
        weightsData.turn = 1;
        weightsData.w0++;
        weightsData.w2--;
        delay(123);
        weightsCharacteristic.writeValue((uint8_t *)&weightsData, sizeof(weightsData), true);
        Serial.println("Updating weights");
      }
    }
}
