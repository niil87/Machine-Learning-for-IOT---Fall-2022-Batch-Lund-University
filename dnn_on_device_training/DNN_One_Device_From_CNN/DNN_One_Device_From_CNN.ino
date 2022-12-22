// Code developed by Nikhil Challa as part of ML in IOT Course - year : 2022
// Team members : Simon Erlandsson

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

#include <ArduinoBLE.h>

#include "cnn_data.h"     // network weights
#include "NN_functions.h" // Neural Network specific functions and definitions





// size of different vectors
size_t numValData = validation_data_cnt;
size_t numTrainData = train_data_cnt;



// to store all the weights and bias for bluetooth transmission
float* WeightBiasPtr;


	









// 0 -> pack vector for bluetooth transmission
// 1 -> unpack vector for updating weights after averaging 
// 2 -> average between values in pointer and location network values, and update both local and pointer value
void packUnpackVector(int Type)
{
  int ptrCount = 0;
  if (Type == 0) {
    // Propagating through network, we store all weights first and then bias. 
    // we start with left most layer, and top most node or lowest to highest index
    for (int i = 1; i < numLayers; i++) {
      for (int j = 0; j < NN_def[i]; j++) {
        for (int k = 0; k < L[i].Neu[j].numInput; k++) {
          WeightBiasPtr[ptrCount] =  L[i].Neu[j].W[k];
          ptrCount += 1;
        }
        WeightBiasPtr[ptrCount] =  L[i].Neu[j].B;
        ptrCount += 1;
      }
    }

    //Serial.print("Total count when packing:");
    //Serial.println(ptrCount);

  } else if (Type == 0) {
    // Propagating through network, we store all weights first and then bias. 
    // we start with left most layer, and top most node or lowest to highest index
    for (int i = 1; i < numLayers; i++) {
      for (int j = 0; j < NN_def[i]; j++) {
        for (int k = 0; k < L[i].Neu[j].numInput; k++) {
           L[i].Neu[j].W[k] = WeightBiasPtr[ptrCount];
          ptrCount += 1;
        }
        L[i].Neu[j].B = WeightBiasPtr[ptrCount];
        ptrCount += 1;
      }
    }
  } else {
    // Propagating through network, we store all weights first and then bias. 
    // we start with left most layer, and top most node or lowest to highest index
    for (int i = 1; i < numLayers; i++) {
      for (int j = 0; j < NN_def[i]; j++) {
        for (int k = 0; k < L[i].Neu[j].numInput; k++) {
          L[i].Neu[j].W[k] = (WeightBiasPtr[ptrCount] + L[i].Neu[j].W[k] )/2;
          WeightBiasPtr[ptrCount] = L[i].Neu[j].W[k];
          ptrCount += 1;
        }
        L[i].Neu[j].B = ( WeightBiasPtr[ptrCount] + L[i].Neu[j].B )/2;
        WeightBiasPtr[ptrCount] = L[i].Neu[j].B;
        ptrCount += 1;
      }
    }
  }

}


void setup() {
	// put your setup code here, to run once:
	Serial.begin(9600);

  // this delay is only for analysis as serial tx is happening before port is ready, 
  // do not use this for demo
	delay(5000);

  // We need to count how many weights and bias we need to transfer
  // the code is only for Fully connected layers
  int NosWeightsBias = CalcTotalWeightsBias();


  Serial.print("The total number of weights and bias:");
  Serial.println(NosWeightsBias);

  WeightBiasPtr = (float*)malloc(NosWeightsBias*sizeof(float));


  

	// randomly initialize the seed based on time
	srand(time(0));
	
  for (int i = 0; i <  numTrainData; i ++ ) {
    indxArray[i] = i;
  }

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
    packUnpackVector(0);
    // send bluetooth
    // do averaging if its leader
    packUnpackVector(2);
    // receive bluetooth
    packUnpackVector(1);

	}

	// checking accuracy if training data
	int correctCount = 0;

  for (int i = 0; i < numTrainData; i++) {
		int maxIndx = 0;
		for (int j = 0; j < IN_VEC_SIZE; j++) {
			input[j] = cnn_train_data[i][j];
		}

		forwardProp();
		for (int j = 1; j < OUT_VEC_SIZE;j++) {
			if (y[maxIndx] < y[j]) {
				maxIndx = j;
			}
		}
		if (maxIndx == train_labels[i]) {
			correctCount+=1;
		}
	}
	
	float Accuracy = correctCount*1.0/numTrainData;
	Serial.print("Training Accuracy:");
	Serial.println(Accuracy);

  correctCount = 0;
	for (int i = 0; i < numValData; i++) {
		int maxIndx = 0;
		for (int j = 0; j < IN_VEC_SIZE; j++) {
			input[j] = cnn_validation_data[i][j];
		}

		forwardProp();
		for (int j = 1; j < OUT_VEC_SIZE;j++) {
			if (y[maxIndx] < y[j]) {
				maxIndx = j;
			}
		}
		if (maxIndx == validation_labels[i]) {
			correctCount+=1;
		}
	}
	
	Accuracy = correctCount*1.0/numValData;
	Serial.print("Validation Accuracy:");
	Serial.println(Accuracy);

}

void loop() {
  // put your main code here, to run repeatedly:

}
