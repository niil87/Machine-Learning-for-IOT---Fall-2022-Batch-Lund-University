// This file was written by Simon Erlandsson, with Nikhil Challa as team member.
// It has been tested and works for 2 devices, but much of the code is generalized
// to allow for more devices, though testing and development of multi-device
// was paused due to lack of time. Runs on Arduino 33 BLE.
#include <ArduinoBLE.h>

#define NBR_BATCHES_ITER (DYN_NBR_WEIGHTS / BLE_NBR_WEIGHTS)

void do_training(); // Is implemented in main-file

typedef struct __attribute__( ( packed ) )
{
  int8_t turn;
  uint8_t batch_id;
  float w[BLE_NBR_WEIGHTS];
} ble_data_t;

/* ------GLOBAL DATA------ */
float* dyn_weights; // Allocated in main
ble_data_t bleData;
BLEDevice peripheral;

BLECharacteristic readCharacteristic;
BLECharacteristic writeCharacteristic;

/* ------DEBUG------ */
#if DEBUG
void PRINT_BLE(bool isSending = true) {
  if (isSending) {
    Serial.print("Sending BLE Values. Turn: ");
  } else {
    Serial.print("Receiving BLE Values. Turn: ");
  }
  Serial.print(bleData.turn);
  Serial.print(", batch_id: ");
  Serial.print(bleData.batch_id);
  Serial.print(", weights: ");
  for (int i = 0; i < BLE_NBR_WEIGHTS; i++) {
    Serial.print(bleData.w[i]);
    Serial.print(" ");
  }
  Serial.println();
}

void PRINT_WEIGHTS() {
  Serial.print("Weights Value: ");
  for (int i = 0; i < DYN_NBR_WEIGHTS; i++) {
    Serial.print(dyn_weights[i]);
    Serial.print(" ");
  }
  Serial.println();
}
#endif

/* ------BLUETOOTH CODE------ */
void loopPeripheral() {
  while (peripheral.connected()) {
    // Has been notified?
    if (readCharacteristic.valueUpdated()) {
      // Fetch from peripheral
      readCharacteristic.readValue((byte *)&bleData, sizeof(bleData));
#if DEBUG
      PRINT_BLE(false);
#endif

      // Is batch, i.e. all new weights before training?
      if (bleData.turn == 1) {
        // Copy weights to local weight vector
        memcpy(dyn_weights + bleData.batch_id * BLE_NBR_WEIGHTS, bleData.w, BLE_NBR_WEIGHTS * sizeof(bleData.w[0]));

        // Calculate our updated weights
        if (bleData.batch_id == NBR_BATCHES_ITER - 1) {
          // Got all data for this iteration
          do_training();
        }
      }

      // Does the leader want our weights?
      if (bleData.turn == CENTRAL_ID && (bleData.batch_id == NBR_BATCHES_ITER - 1 || CENTRAL_ID != 1)) {
        bleData.turn = 0;

        // Should also be sent in batch
        for (int i = 0; i < NBR_BATCHES_ITER; i++) {
          bleData.batch_id = i;
          // Copy weights
          memcpy(bleData.w, dyn_weights + i * BLE_NBR_WEIGHTS, BLE_NBR_WEIGHTS * sizeof(bleData.w[0]));
          writeCharacteristic.writeValue((byte *)&bleData, sizeof(bleData));
#if DEBUG
          PRINT_BLE(true);
#endif
        }
      }
    }
  }
}

void connectPeripheral() {
  // connect to the peripheral
#if DEBUG
  Serial.println("Connecting ...");
#endif

  if (peripheral.connect()) {
#if DEBUG
    Serial.println("Connected");
#endif
  } else {
#if DEBUG
    Serial.println("Failed to connect!");
#endif
    return;
  }

  // discover peripheral attributes
  Serial.println("Discovering attributes ...");
  if (peripheral.discoverAttributes()) {
#if DEBUG
    Serial.println("Attributes discovered");
#endif
  } else {
#if DEBUG
    Serial.println("Attribute discovery failed!");
#endif
    peripheral.disconnect();
    return;
  }

  // retrieve the Weights characteristics
  readCharacteristic = peripheral.characteristic("19b10001-e8f2-537e-4f6c-d104768a1214");
  writeCharacteristic = peripheral.characteristic("19b10001-e8f2-537e-4f6c-d104768a1215");

  if (!readCharacteristic || !writeCharacteristic) {
#if DEBUG
    Serial.println("Peripheral does not have WeightsCharacteristic characteristic!");
#endif
    peripheral.disconnect();
    return;
  }
  if (!readCharacteristic.subscribe()) {
#if DEBUG
    Serial.println("Subscription failed!");
#endif
    peripheral.disconnect();
    return;
  }

  // Inform peripheral, connection is established
  bleData.turn = -1;
  writeCharacteristic.writeValue((byte *)&bleData, sizeof(bleData));

  // Continues until disconnect
  loopPeripheral();

  Serial.println("Peripheral disconnected");
}

void setupBLE(DATA_TYPE* wbptr) {
  dyn_weights = (float*) wbptr;    // we only support float for BLE transmission
  // initialize the BLE hardware
  BLE.begin();

#if DEBUG
  Serial.println("BLE Central - Weights control Setup Done");
#endif

  // start scanning for peripherals
  BLE.scanForUuid("19b10000-e8f2-537e-4f6c-d104768a1214");
}

void loopBLE() {
  // check if a peripheral has been discovered
  peripheral = BLE.available();

  if (peripheral) {
    // discovered a peripheral, print out address, local name, and advertised service
#if DEBUG
    Serial.print("Found ");
    Serial.print(peripheral.address());
    Serial.print(" '");
    Serial.print(peripheral.localName());
    Serial.print("' ");
    Serial.print(peripheral.advertisedServiceUuid());
    Serial.println();
#endif

    if (peripheral.localName() != "MLLeader") {
      return;
    }

    // stop scanning
    BLE.stopScan();

    connectPeripheral();

    // peripheral disconnected, start scanning again
    BLE.scanForUuid("19b10000-e8f2-537e-4f6c-d104768a1214");
  }
}
