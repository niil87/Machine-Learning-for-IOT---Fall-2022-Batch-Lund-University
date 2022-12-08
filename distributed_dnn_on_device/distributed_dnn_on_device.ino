// Code developed by Nikhil Challa as part of ML in IOT Course - year : 2022
// Team members : Simon Erlandsson

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

#include <ArduinoBLE.h>

#include "cnn_data.h"     // network weights

// NN parameters
#define LEARNING_RATE 0.01
#define EPOCH 25

// DO NOT TOUCH THE FIRST AND LAST ENTRIES OF BELOW ARRAY, YOU CAN MODIFY ANY OF OTHER ENTRIES
// like increase the number of layers, change the nodes per layer
static const int NN_def[] = {first_layer_input_cnt, 20, classes_cnt};

// size of different vectors
size_t numValData = validation_data_cnt;
size_t numTrainData = train_data_cnt;

#include "NN_functions.h" // Neural Network specific functions and definitions

// to store all the weights and bias for bluetooth transmission
float* WeightBiasPtr;


void setup() {
	// put your setup code here, to run once:
	Serial.begin(9600);

  // this delay is only for analysis as serial tx is happening before port is ready, 
  // do not use this for demo
	delay(5000);

  // We need to count how many weights and bias we need to transfer
  // the code is only for Fully connected layers
  int NosWeightsBias = calcTotalWeightsBias();

  Serial.print("The total number of weights and bias:");
  Serial.println(NosWeightsBias);

  WeightBiasPtr = (float*)malloc(NosWeightsBias*sizeof(float));


  

	// randomly initialize the seed based on time
	srand(time(0));

	createNetwork();
 
	for (int i = 0; i < EPOCH; i++) {
		Serial.print("Epoch count:");
		Serial.print(i);
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
    packUnpackVector(0,WeightBiasPtr);
    // send bluetooth
    // do averaging if its master
    packUnpackVector(2,WeightBiasPtr);
    // receive bluetooth
    packUnpackVector(1,WeightBiasPtr);

	}

  printAccuracy();

}

void loop() {
  // put your main code here, to run repeatedly:

}
