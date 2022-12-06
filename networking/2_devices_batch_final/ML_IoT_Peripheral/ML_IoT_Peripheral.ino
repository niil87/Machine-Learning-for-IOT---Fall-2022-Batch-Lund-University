#include <ArduinoBLE.h>

#define NBR_CENTRALS 1
#define DYN_NBR_WEIGHTS 3600 // MUST BE MULTIPLE OF BLE_NBR_WEIGHTS!
#define BLE_NBR_WEIGHTS 24
#define NBR_BATCHES_ITER (DYN_NBR_WEIGHTS / BLE_NBR_WEIGHTS) // Max 255 before overload

#define DEBUG 0

typedef struct __attribute__( ( packed ) )
{
  int8_t turn;
  uint8_t batch_id;
  int16_t w[BLE_NBR_WEIGHTS];
} ble_data_t;

/* ------GLOBAL DATA------ */
int16_t dyn_weights[DYN_NBR_WEIGHTS];
ble_data_t bleData;
int iter_cnt = 0;

BLEService weightsService("19B10000-E8F2-537E-4F6C-D104768A1214");  // BLE Weights Service

BLECharacteristic readCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLEIndicate, sizeof(bleData));
BLECharacteristic writeCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1215", BLEWrite, sizeof(bleData));

const int ledPin = LED_BUILTIN;  // pin to use for the LED

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

/* ------ML CODE------ */
void aggregate_weights() {
#if DEBUG
  Serial.println("Aggregating");
  PRINT_BLE(false);
#endif
  // Read input, does not need to be stored in a buffer, just use .value() or [] to access offset;
  // weightsData = *((weights_data_t *)writeCharacteristic.value());
  // But for now we copied the data into "bleData" before call to this function, might optimize later
  for (int i = 0; i < BLE_NBR_WEIGHTS; i++) {
    int idx = (signed int) bleData.batch_id * BLE_NBR_WEIGHTS + i;
    dyn_weights[idx] += bleData.w[i];
  }

  //delay(30);
}

void do_training() {
  dyn_weights[0]++;
  dyn_weights[DYN_NBR_WEIGHTS - 1]++;
  dyn_weights[3]++;

#if DEBUG
  Serial.println("Peripheral Training");
  PRINT_WEIGHTS();
#endif

  //delay(250);
}

/* ------BLUETOOTH CODE------ */
void ConnectHandler(BLEDevice central) {
  // central connected event handler
#if DEBUG
  Serial.print("Connected event, central: ");
  Serial.println(central.address());
#endif
  BLE.advertise();
}

void DisconnectHandler(BLEDevice central) {
  // central disconnected event handler
#if DEBUG
  Serial.print("Disconnected event, central: ");
  Serial.println(central.address());
#endif
  BLE.advertise();
}

int turn = 1;

void setup() {
  Serial.begin(9600);
  //while (!Serial);

  // set LED pin to output mode
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, HIGH);

  // begin initialization
  if (!BLE.begin()) {
#if DEBUG
    Serial.println("starting BLE failed!");
    while (1)
      ;
#endif
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

  // Client should respond
  writeCharacteristic.writeValue((byte *)&bleData, sizeof(bleData));

  bleData.turn = turn;
  bleData.batch_id == NBR_BATCHES_ITER - 1;
  readCharacteristic.writeValue((byte *)&bleData, sizeof(bleData));

  // start advertising
  BLE.advertise();

#if DEBUG
  Serial.println("BLE Weights Peripheral");
#endif
}

unsigned long old_time;

void loop() {
  unsigned long current_time = millis();
  BLE.poll();
  bool timeout = current_time - old_time > 10000;
  if (writeCharacteristic.written() || timeout) {
    old_time = current_time;
    int8_t receivedTurn = writeCharacteristic[0];

    // Master needs to do something
    if (receivedTurn == 0 || timeout) {
      uint8_t batch_id = writeCharacteristic[1];
      // Add the received weights to our weight vector
      writeCharacteristic.readValue((byte *)&bleData, sizeof(bleData));
      aggregate_weights();

      // Have all weights for this iteration been aggregated? I.e. iteration finished?
      if (turn == NBR_CENTRALS && batch_id == NBR_BATCHES_ITER - 1 || timeout) {
        turn = 1; // Start over
        bleData.turn = turn;
        // Beginning of new iteration, distribute weights from previous iteration and tell node 1 to send training result

        Serial.print("-- STARTING ITERATION: ");
        Serial.print(iter_cnt++);
        Serial.println(" --");

        // Send multiple batches (all weights)
        for (int i = 0; i < NBR_BATCHES_ITER; i++) {
          bleData.batch_id = i;
          memcpy(bleData.w, dyn_weights + i * BLE_NBR_WEIGHTS, BLE_NBR_WEIGHTS * sizeof(bleData.w[0]));
          readCharacteristic.writeValue((byte *)&bleData, sizeof(bleData));
#if DEBUG
          PRINT_BLE(true);
#endif
        }

        // Calculate our contribution to next iteration, while others also are calculating
        // But dont if timeout, to avoid unequal sample size
        if (!timeout) do_training();
      } else if (bleData.batch_id == NBR_BATCHES_ITER - 1 ) {
        bleData.turn = ++turn;
        // Tell next node to send its weights, note: no need to redistribute weights
        readCharacteristic.writeValue((byte *)&bleData, sizeof(bleData));
#if DEBUG
        PRINT_BLE(true);
#endif
      }
    }
  }
}
