#include <ArduinoBLE.h>

#define NBR_CENTRALS 2
#define NBR_WEIGHTS 24

typedef struct __attribute__( ( packed ) )
{
  uint8_t turn;
  int16_t w[NBR_WEIGHTS];
} weights_data_t;

weights_data_t weightsData;

BLEService weightsService("19B10000-E8F2-537E-4F6C-D104768A1214");  // BLE Weights Service

BLECharacteristic readCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify, sizeof(weightsData));
BLECharacteristic writeCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1215", BLEWrite, sizeof(weightsData));

const int ledPin = LED_BUILTIN;  // pin to use for the LED

void ConnectHandler(BLEDevice central) {
  // central connected event handler
  Serial.print("Connected event, central: ");
  Serial.println(central.address());
  BLE.advertise();
  readCharacteristic.writeValue((uint8_t *)&weightsData, sizeof(weightsData));
  
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
  weightsService.addCharacteristic(readCharacteristic);
  weightsService.addCharacteristic(writeCharacteristic);

  // add service
  BLE.addService(weightsService);
  // set the initial value for the characeristic:
  weightsData.turn = 1;
  readCharacteristic.writeValue((uint8_t *)&weightsData, sizeof(weightsData));
  writeCharacteristic.writeValue((uint8_t *)&weightsData, sizeof(weightsData));

  // start advertising
  BLE.advertise();

  Serial.println("BLE Weights Peripheral");
}

int lastTurn = NBR_CENTRALS;
unsigned long old_time;

void loop() {
  unsigned long current_time = millis();
 
  BLE.poll();
      if (writeCharacteristic.written() || current_time - old_time > 1000) {
      old_time = current_time;
      // Below can be used to read new weights and applying them to current weights, for now we just copy
      weightsData = *((weights_data_t *)writeCharacteristic.value());
      //weightsCharacteristic.readValue((uint8_t *)&weightsData, sizeof(weightsData)); // Also works, but copies directly
      Serial.print("Peripheral Weights Value: ");
      Serial.print(weightsData.turn);
      Serial.print(" ");
      Serial.print(weightsData.w[0]);
      Serial.print(" ");
      Serial.print(weightsData.w[1]);
      Serial.print(" ");
      Serial.print(weightsData.w[2]);
      Serial.print(" ");
      Serial.println(weightsData.w[NBR_WEIGHTS - 1]);
  
      if (weightsData.turn == 0) {
        weightsData.turn = (++lastTurn - 1) % NBR_CENTRALS + 1;
        weightsData.w[1]++;
        weightsData.w[2]--;
        weightsData.w[NBR_WEIGHTS - 1]++;
        //delay(123);
        readCharacteristic.writeValue((uint8_t *)&weightsData, sizeof(weightsData), true);
        Serial.println("Updating weights");
      }
    }
}
