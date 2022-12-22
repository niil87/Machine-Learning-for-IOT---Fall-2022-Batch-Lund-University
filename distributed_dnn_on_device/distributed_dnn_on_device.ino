// Code developed by Nikhil Challa and Simon Erlandsson as part of ML in IOT Course - year : 2022

#define LEADER 1
#define WORKER 2

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int iter_cnt = 0;
int weights_bias_cnt = 0;
extern const int first_layer_input_cnt;
extern const int classes_cnt;

/* ------- CONFIG ------- */
#define DEVICE_TYPE WORKER // Which device is being exported: LEADER or WORKER?
#define DEBUG 0

/*
 *  Should LEADER and WORKER use different datasets? Ex. one label is only present in the training set
 *  for one of the devices. Makes the distribution more obvious.
 */
#define USE_BIASED_DATASETS 1

/*
 *  Should the incoming weights be saved (and taken into account) or ignored? Can be used to
 *  test the accuracy if only training locally, using the device's own data.
 *  BLE is still used, to synchronize between iterations, thus can also be used to test BLE connection.
 *  NOTE: ENABLE_BLE should be set to 1 if this is enabled.
 */
#define USE_DISTRIBUTED_WEIGHTS 1

/*
 *  Disabling BLE runs the device completely locally. I.e. training happens as if
 *  USE_DISTRIBUTED_WEIGHTS = 0 and no synchronization is done at all.
 *  Use for testing on one device.
 */
#define ENABLE_BLE 1

/* 
 *  DYN_NBR_WEIGHTS defines how many weights that should be transfered via BLE in total/iteration.
 *  This is calculated at runtime now. Note that it has to be a multiple of BLE_NR_WEIGHTS
 *  and that DYN_NBR_WEIGHTS / BLE_NBR_WEIGHTS < 256 to prevent overflow.
 */
#define DYN_NBR_WEIGHTS weights_bias_cnt // MUST BE MULTIPLE OF BLE_NBR_WEIGHTS!

/*
 *  For BLE_NR_WEIGHTS, i.e. weights per BLE-transmission, we had big reliability issues
 *  if the payload was > 50 bytes per package. In the struct we use, two bytes are used
 *  for synchronization, leaving 48 bytes to weights. If using floats (as we do), this gives
 *  12 weights per BLE-transmission.
 */
#define BLE_NBR_WEIGHTS 12

// NN parameters
#define LEARNING_RATE 0.01
#define EPOCH 50 

// DO NOT TOUCH THE FIRST AND LAST ENTRIES OF BELOW ARRAY, YOU CAN MODIFY ANY OF OTHER ENTRIES
// like increase the number of layers, change the nodes per layer
static const int NN_def[] = {first_layer_input_cnt, 20, classes_cnt};

// this is to set the precision for weight/bias in NN
#define DATA_TYPE_FLOAT  // Valid values:  DATA_TYPE_DOUBLE , DATA_TYPE_FLOAT

/* ------- END CONFIG ------- */

// Training and Validation data
#if USE_BIASED_DATASETS && DEVICE_TYPE == LEADER
#include "cnn_data_biased_1.h"  
#elif USE_BIASED_DATASETS && DEVICE_TYPE == WORKER
#include "cnn_data_biased_2.h"
#else
#include "cnn_data.h"  
#endif
    
#include "NN_functions.h" // Neural Network specific functions and definitions

#if DEVICE_TYPE == LEADER
#define NBR_CENTRALS 1 // Config
#include "BLE_peripheral.h"

#elif DEVICE_TYPE == WORKER
#define CENTRAL_ID 1 // Config
#include "BLE_central.h"
#endif

void destroy() {
  Serial.println("Finished training, shutting down.");
  printAccuracy();
#if ENABLE_BLE
  BLE.stopAdvertise();
  BLE.disconnect();
#endif
  while (1) ;
}

/* Called by BLE for LEADER when all weights from WORKER(s) have been received */
#if DEVICE_TYPE == LEADER
void aggregate_weights() {
#if DEBUG
  Serial.println("Aggregating");
#endif

  Serial.println("Accuracy before aggregation:");
  printAccuracy();

#if USE_DISTRIBUTED_WEIGHTS && ENABLE_BLE
  // Merge incoming weights with leader stored weights
  // Also exports the new weights to worker(s)
  packUnpackVector(AVERAGE);
#endif

  Serial.println("Accuracy after aggregation:");
  printAccuracy();
}
#endif

/* Called by BLE when all weights have been received/aggregated */
void do_training() {
#if DEVICE_TYPE == LEADER || !ENABLE_BLE
  if (iter_cnt >= EPOCH) destroy();
#endif

// Worker has to unpack from BLE while this is done in the aggregation for the leader
#if DEVICE_TYPE == WORKER && USE_DISTRIBUTED_WEIGHTS && ENABLE_BLE
  packUnpackVector(UNPACK);
  Serial.println("Accuracy using incoming weights:");
  printAccuracy();
#endif

#if DEBUG
  Serial.println("Now Training");
  PRINT_WEIGHTS();
#endif

  Serial.print("Epoch count (training count): ");
  Serial.print(++iter_cnt);
  Serial.println();

  // reordering the index for more randomness and faster learning
  shuffleIndx();
  
  // starting forward + Backward propagation
  for (int j = 0;j < numTrainData;j++) {
    generateTrainVectors(j);  
    forwardProp();
    backwardProp();
  }

  // pack the vector for bluetooth transmission
  forwardProp();
#if DEVICE_TYPE == WORKER || !ENABLE_BLE
#if USE_DISTRIBUTED_WEIGHTS && ENABLE_BLE
  packUnpackVector(PACK);
  Serial.println("Accuracy after local training:");
#endif
  printAccuracy();
#endif
}

void setup() {
  // randomly initialize the seed based on time
  srand(time(0));

  Serial.begin(9600);
  delay(5000);

// unsupported config handling
#ifdef DATA_TYPE_DOUBLE && USE_DISTRIBUTED_WEIGHTS
  Serial.println("Sorry, we dont support DOUBLE precision with distributed learning, only FLOAT is allowed");
  Serial.println("You can use DOUBLE for single device training");
  exit(0);
#endif

  // We need to count how many weights and bias we need to transfer
  // the code is only for Fully connected layers
  weights_bias_cnt = calcTotalWeightsBias();

#if !USE_DISTRIBUTED_WEIGHTS
  weights_bias_cnt = BLE_NBR_WEIGHTS; // Since weights are not being used, only a packet for synchronization is needed
#endif

  // weights_bias_cnt has to be multiple of BLE_NBR_WEIGHTS
  int remainder = weights_bias_cnt % BLE_NBR_WEIGHTS;
  if (remainder != 0)
    weights_bias_cnt += BLE_NBR_WEIGHTS - remainder;

  Serial.print("The total number of weights and bias used for BLE: ");
  Serial.println(weights_bias_cnt);

  // Allocate common weight vector, and pass to setupNN, setupBLE
  DATA_TYPE* WeightBiasPtr = (DATA_TYPE*) calloc(weights_bias_cnt, sizeof(DATA_TYPE));

  setupNN(WeightBiasPtr);  // CREATES THE NETWORK BASED ON NN_def[]
  printAccuracy();
#if ENABLE_BLE
  setupBLE(WeightBiasPtr);
#endif
}

void loop() {
  // Note that for the "central": another loop is ran
  // when connceted to a peripheral. This function is
  // then not reachable.
#if ENABLE_BLE
  loopBLE();
#else
  do_training(); // Local training
#endif
}
