// This file was written by Simon Erlandsson, with Nikhil Challa as team member.
#include <ArduinoBLE.h>

#define CENTRAL_ID 2
#define NBR_WEIGHTS 24

typedef struct __attribute__( ( packed ) )
{
  uint8_t turn;
  int16_t w[NBR_WEIGHTS];
} weights_data_t;

weights_data_t weightsData;

unsigned long last_read_time = 0;

void listenWeights(BLEDevice peripheral) {
  // connect to the peripheral
  Serial.println("Connecting ...");

  if (peripheral.connect()) {
    Serial.println("Connected");
  } else {
    Serial.println("Failed to connect!");
    return;
  }

  // discover peripheral attributes
  Serial.println("Discovering attributes ...");
  if (peripheral.discoverAttributes()) {
    Serial.println("Attributes discovered");
  } else {
    Serial.println("Attribute discovery failed!");
    peripheral.disconnect();
    return;
  }

  // retrieve the Weights characteristics
  BLECharacteristic readCharacteristic = peripheral.characteristic("19b10001-e8f2-537e-4f6c-d104768a1214");
  BLECharacteristic writeCharacteristic = peripheral.characteristic("19b10001-e8f2-537e-4f6c-d104768a1215");

  if (!readCharacteristic || !writeCharacteristic) {
    Serial.println("Peripheral does not have WeightsCharacteristic characteristic!");
    peripheral.disconnect();
    return;
  }
  if (!readCharacteristic.subscribe()) {
    Serial.println("Subscription failed!");
    peripheral.disconnect();
    return;
  }
  while (peripheral.connected()) {
    unsigned long now = millis();
    if (readCharacteristic.valueUpdated() || now - last_read_time > 500) {
      last_read_time = now;
      readCharacteristic.readValue((uint8_t *)&weightsData, sizeof(weightsData));
      Serial.print("Central Weights Value: ");
      Serial.print(weightsData.turn);
      Serial.print(" ");
      Serial.print(weightsData.w[0]);
      Serial.print(" ");
      Serial.print(weightsData.w[1]);
      Serial.print(" ");
      Serial.print(weightsData.w[2]);
      Serial.print(" ");
      Serial.println(weightsData.w[NBR_WEIGHTS - 1]);
      if (weightsData.turn == CENTRAL_ID) {
        weightsData.turn = 0;
  
        Serial.println("Updating weights and sending to master");
        weightsData.w[0]++;
        weightsData.w[1]++;
        weightsData.w[2]++;
        weightsData.w[NBR_WEIGHTS - 1]++;
        //delay(185);
        writeCharacteristic.writeValue((uint8_t *)&weightsData, sizeof(weightsData), true);
      }
    }
  }
  Serial.println("Peripheral disconnected");
}

void setup() {
  Serial.begin(9600);
  //while (!Serial);

  // initialize the BLE hardware
  BLE.begin();

  Serial.println("BLE Central - Weights control");

  // start scanning for peripherals
  BLE.scanForUuid("19b10000-e8f2-537e-4f6c-d104768a1214");
}

void loop() {
  // check if a peripheral has been discovered
  BLEDevice peripheral = BLE.available();

  if (peripheral) {
    // discovered a peripheral, print out address, local name, and advertised service
    Serial.print("Found ");
    Serial.print(peripheral.address());
    Serial.print(" '");
    Serial.print(peripheral.localName());
    Serial.print("' ");
    Serial.print(peripheral.advertisedServiceUuid());
    Serial.println();

    if (peripheral.localName() != "MLMaster") {
      return;
    }

    // stop scanning
    BLE.stopScan();

    listenWeights(peripheral);

    // peripheral disconnected, start scanning again
    BLE.scanForUuid("19b10000-e8f2-537e-4f6c-d104768a1214");
  }
}
