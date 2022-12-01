#include <ArduinoBLE.h>

#define CENTRAL_ID 2
#define NBR_WEIGHTS 24

typedef struct __attribute__( ( packed ) )
{
  uint8_t turn;
  uint8_t batch_id;
  int16_t w[NBR_WEIGHTS];
} weights_data_t;

/* ------GLOBAL DATA------ */
weights_data_t weightsData;
BLEDevice peripheral;

BLECharacteristic readCharacteristic;
BLECharacteristic writeCharacteristic;

/* ------ML CODE------ */
void do_training() {
  Serial.println("Training");

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

  weightsData.w[0] = 1;
  weightsData.w[CENTRAL_ID] = 1;
  weightsData.w[NBR_WEIGHTS - 1] = 1;
  Serial.println("Updating weights and sending to master");
  //delay(250);
}

/* ------BLUETOOTH CODE------ */
unsigned long last_read_time = 0;

void loopPeripheral() {
  while (peripheral.connected()) {
    unsigned long now = millis();
    // Has been notified, or timeout?
    if (readCharacteristic.valueUpdated() || now - last_read_time > 10000) {
      last_read_time = now;

      // Fetch from peripheral
      readCharacteristic.read();

      // Read first byte to see turn in message (message type)
      uint8_t receivedTurn = readCharacteristic[0];
      uint8_t batch_id = readCharacteristic[1];
      Serial.println(batch_id);

      // Is batch, i.e. all new weights before training?
      if (receivedTurn == 1) {
        // Copy weights (for now not batch), can also use [] to avoid copy
        //readCharacteristic.readValue((byte *)&weightsData, sizeof(weightsData));

        // Calculate our updated weights
        do_training();
      }

      // Does the master want our weights?
      if (receivedTurn == CENTRAL_ID) {
        weightsData.turn = 0;

        // Should also be sent in batch
        writeCharacteristic.writeValue((byte *)&weightsData, sizeof(weightsData));
      }
    }
  }
}

void connectPeripheral() {
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
  readCharacteristic = peripheral.characteristic("19b10001-e8f2-537e-4f6c-d104768a1214");
  writeCharacteristic = peripheral.characteristic("19b10001-e8f2-537e-4f6c-d104768a1215");

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

  // Continues until disconnect
  loopPeripheral();

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
  peripheral = BLE.available();

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

    connectPeripheral();

    // peripheral disconnected, start scanning again
    BLE.scanForUuid("19b10000-e8f2-537e-4f6c-d104768a1214");
  }
}
