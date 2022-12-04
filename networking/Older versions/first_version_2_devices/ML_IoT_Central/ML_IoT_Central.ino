#include <ArduinoBLE.h>

typedef struct __attribute__( ( packed ) )
{
  uint8_t turn;
  int16_t w0;
  int16_t w1;
  int16_t w2;
} weights_data_t;

weights_data_t weightsData;

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

  // retrieve the Weights characteristic
  BLECharacteristic weightsCharacteristic = peripheral.characteristic("19b10001-e8f2-537e-4f6c-d104768a1214");

  if (!weightsCharacteristic) {
    Serial.println("Peripheral does not have WeightsCharacteristic characteristic!");
    peripheral.disconnect();
    return;
  }
  if (!weightsCharacteristic.subscribe()) {
    Serial.println("Subscription failed!");
    peripheral.disconnect();
    return;
  }
  while (peripheral.connected()) {
    //if (weightsCharacteristic.valueUpdated()) { // Seems to not be working with both read/write at the same time
      weightsCharacteristic.readValue((uint8_t *)&weightsData, sizeof(weightsData));
      Serial.print("Central Weights Value: ");
      Serial.print(weightsData.turn);
      Serial.print(" ");
      Serial.print(weightsData.w0);
      Serial.print(" ");
      Serial.print(weightsData.w1);
      Serial.print(" ");
      Serial.println(weightsData.w2);
      if (weightsData.turn == 1) {
        weightsData.turn = 0;
  
        Serial.println("Updating weights and sending to master");
        weightsData.w1++;
        weightsData.w2++;
        delay(185);
        weightsCharacteristic.writeValue((uint8_t *)&weightsData, sizeof(weightsData), true);
      }
  //}
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
