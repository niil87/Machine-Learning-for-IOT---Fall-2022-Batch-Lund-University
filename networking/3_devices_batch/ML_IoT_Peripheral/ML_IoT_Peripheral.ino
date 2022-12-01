#include <ArduinoBLE.h>

#define NBR_CENTRALS 2
#define NBR_WEIGHTS 24

typedef struct __attribute__( ( packed ) )
{
  uint8_t turn;
  uint8_t batch_id;
  int16_t w[NBR_WEIGHTS];
} weights_data_t;

/* ------GLOBAL DATA------ */
weights_data_t weightsData;

BLEService weightsService("19B10000-E8F2-537E-4F6C-D104768A1214");  // BLE Weights Service

BLECharacteristic readCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLEIndicate, sizeof(weightsData));
BLECharacteristic writeCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1215", BLEWrite, sizeof(weightsData));

const int ledPin = LED_BUILTIN;  // pin to use for the LED

/* ------ML CODE------ */
void aggregate_weights() {
  Serial.println("Aggregating");

  // Read input, does not need to be stored in a buffer, just use .value() or [] to access offset;
  //weightsData = *((weights_data_t *)writeCharacteristic.value());
  for (uint8_t i = 0; i < NBR_WEIGHTS; i++) {
    weightsData.w[i] += (*(weights_data_t *)writeCharacteristic.value()).w[i];
  }

  Serial.print("Peripheral Weights Value: ");
  Serial.print(weightsData.turn);
  Serial.print(" ");
  Serial.print(weightsData.w[0]);
  Serial.print(" ");
  Serial.print(weightsData.w[1]);
  Serial.print(" ");
  Serial.print(weightsData.w[2]);
  Serial.print(" ");
  Serial.print(weightsData.w[3]);
  Serial.print(" ");
  Serial.print(weightsData.w[4]);
  Serial.print(" ");
  Serial.println(weightsData.w[NBR_WEIGHTS - 1]);
  //delay(30);
}

void do_training() {
  Serial.println("Training");

  weightsData.w[0]++;
  weightsData.w[NBR_WEIGHTS - 1]++;
    weightsData.w[3]++;

  
  Serial.print("Peripheral Weights Value: ");
  Serial.print(weightsData.turn);
  Serial.print(" ");
  Serial.print(weightsData.w[0]);
  Serial.print(" ");
  Serial.print(weightsData.w[1]);
  Serial.print(" ");
  Serial.print(weightsData.w[2]);
  Serial.print(" ");
  Serial.print(weightsData.w[3]);
  Serial.print(" ");
  Serial.print(weightsData.w[4]);
  Serial.print(" ");
  Serial.println(weightsData.w[NBR_WEIGHTS - 1]);
  //delay(250);
  //aggregate_weights();
}

/* ------BLUETOOTH CODE------ */

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

int turn = 1;

void setup() {
  Serial.begin(9600);
  while (!Serial);

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
  weightsData.turn = turn;
  readCharacteristic.writeValue((byte *)&weightsData, sizeof(weightsData));
  writeCharacteristic.writeValue((byte *)&weightsData, sizeof(weightsData));

  // start advertising
  BLE.advertise();

  Serial.println("BLE Weights Peripheral");
}

unsigned long old_time;

void loop() {
  unsigned long current_time = millis();
  BLE.poll();
  if (writeCharacteristic.written() || current_time - old_time > 10000) {
    old_time = current_time;
    uint8_t receivedTurn = writeCharacteristic[0];

    // Master needs to do something
    if (receivedTurn == 0) {
      // Add the received weights to our weight vector
      aggregate_weights();

      // Have all weights for this iteration been aggregated? I.e. iteration finished?
      if (turn == NBR_CENTRALS) {
        turn = 1; // Start over
        weightsData.turn = turn;
        // Beginning of new iteration, distribute weights from previous iteration and tell node 1 to send training result
        // Same for now, but should be multi batch
        weightsData.batch_id = 255;
        readCharacteristic.writeValue((byte *)&weightsData, sizeof(weightsData));

        // Calculate our contribution to next iteration, while others also are calculating
        do_training();
      } else {
        weightsData.turn = ++turn;
        // Tell next node to send its weights, note: no need to redistribute weights
        readCharacteristic.writeValue((byte *)&weightsData, sizeof(weightsData));
      }
    }
  }
}
