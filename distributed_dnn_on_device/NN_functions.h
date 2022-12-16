// Code developed by Nikhil Challa as part of ML in IOT Course - year : 2022
// Team members : Simon Erlandsson
// To get full understanding of the code, please refer to below document in git
// https://github.com/niil87/Machine-Learning-for-IOT---Fall-2022-Batch-Lund-University/blob/main/Math_for_Understanding_Deep_Neural_Networks.pdf
// The equations that corresponds to specific code will be listed in the comments next to code

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define fRAND ( rand()*1.0/RAND_MAX-0.5 )*2   // random number generator between -1 and +1 
#define ACT(a) MAX(a,0)    // RELU(a)



#ifdef DATA_TYPE_FLOAT 
  #define DATA_TYPE float
  #define EXP_LIMIT 78.0  // limit 88.xx but we need to factor in accumulation for softmax
  #define EXP(a) expl(a)
#else
  #define DATA_TYPE double
  #define EXP_LIMIT 699.0 // limit is 709.xx but we need to factor in accumulation for softmax
  #define EXP(a) exp(a)
#endif

#define IN_VEC_SIZE first_layer_input_cnt
#define OUT_VEC_SIZE classes_cnt

// size of different vectors
size_t numTestData = test_data_cnt;
size_t numValData = validation_data_cnt;
size_t numTrainData = train_data_cnt;


size_t numLayers = sizeof(NN_def) / sizeof(NN_def[0]);
// size of the input to NN

// dummy input for testing
DATA_TYPE input[IN_VEC_SIZE];

// dummy output for testing
DATA_TYPE hat_y[OUT_VEC_SIZE];    // target output
DATA_TYPE y[OUT_VEC_SIZE];        // output after forward propagation


// creating array index to randomnize order of training data
int indxArray[train_data_cnt];

// Convention: Refer to 
typedef struct neuron_t {
	int numInput;
	DATA_TYPE* W;
	DATA_TYPE B;
	DATA_TYPE X;

	// For back propagation, convention, dA means dL/dA or partial derivative of Loss over Accumulative output
	DATA_TYPE* dW;
	DATA_TYPE dA;
	DATA_TYPE dB;

} neuron;

typedef struct layer_t {
	int numNeuron;
	neuron* Neu;
} layer;

// initializing the layer as global parameter
layer* L = NULL;

// Weights written to here will be sent/received via bluetooth. 
DATA_TYPE* WeightBiasPtr = NULL;

// Equation (8)
DATA_TYPE AccFunction (int layerIndx, int nodeIndx) {
	DATA_TYPE A = 0;

	for (int k = 0; k < NN_def[layerIndx - 1]; k++) {

	// updating weights/bais and resetting gradient value if non-zero
	if (L[layerIndx].Neu[nodeIndx].dW[k] != 0.0 ) {
		L[layerIndx].Neu[nodeIndx].W[k] += L[layerIndx].Neu[nodeIndx].dW[k];
		L[layerIndx].Neu[nodeIndx].dW[k] = 0.0;
	}

	A += L[layerIndx].Neu[nodeIndx].W[k] * L[layerIndx - 1].Neu[k].X;

	}

	if (L[layerIndx].Neu[nodeIndx].dB != 0.0 ) {
		L[layerIndx].Neu[nodeIndx].B += L[layerIndx].Neu[nodeIndx].dB;
		L[layerIndx].Neu[nodeIndx].dB = 0.0;
	}
	A += L[layerIndx].Neu[nodeIndx].B;

	return A;
}


// NEED HANDLING TO ENSURE NO WEIGHTS AND BIAS ARE CREATED FOR FIRST LAYER OR THROW ERROR IF ACCESSED ACCIDENTLY
// EVEN THOUGH WE HAVENT EXPLICITLY CALLED CREATED THE NEURON FOR FIRST LAYER, HOW WAS L[i].Neu[j].X SUCCESSFUL IN FORWARD PROPAGATION!!
neuron createNeuron(int numInput) {

	neuron N1;

	N1.B = fRAND;
	N1.numInput = numInput;
	N1.W = (DATA_TYPE*)calloc(numInput, sizeof(DATA_TYPE));
	N1.dW = (DATA_TYPE*)calloc(numInput, sizeof(DATA_TYPE));
	// initializing values of W to rand and dW to 0
	int Sum = 0;
	for (int i = 0; i < numInput; i++) {
		N1.W[i] = fRAND;
		N1.dW[i] = 0.0;
	}
	N1.dA = 0.0;
	N1.dB = 0.0;

	return N1;

}

layer createLayer (int numNeuron) {
	layer L1;
	L1.numNeuron = numNeuron;
	L1.Neu = (neuron*)calloc(numNeuron, sizeof(neuron));
	return L1;
}

void createNetwork() {

	L = (layer*)calloc(numLayers, sizeof(layer));

	// First layer has no input weights
	L[0] = createLayer(NN_def[0]);

	for (int i = 1; i < numLayers; i++) {
		L[i] = createLayer(NN_def[i]);
		for (int j = 0; j < NN_def[i]; j++) {
			L[i].Neu[j] = createNeuron(NN_def[i - 1]);
		}
	}

	// creating indx array for shuffle function to be used later
	for (int i = 0; i <  numTrainData; i ++ ) {
		indxArray[i] = i;
	}

}


// this function is to calculate dA
DATA_TYPE dLossCalc( int layerIndx, int nodeIndx) {

	DATA_TYPE Sum = 0;
	int outputSize = NN_def[numLayers - 1];
	// for the last layer, we use complex computation
	if (layerIndx == numLayers - 1) {	
		Sum = y[nodeIndx] - hat_y[nodeIndx];										// Equation (17)
	// for all except last layer, we use simple aggregate of dA
	} else if (AccFunction(layerIndx, nodeIndx) > 0)  {   							
		for (int i = 0; i < NN_def[layerIndx + 1]; i++) {
			Sum += L[layerIndx + 1].Neu[i].dA * L[layerIndx + 1].Neu[i].W[nodeIndx]; 	// Equation (24)
		}
	} else {   																		// refer to "Neat Trick" and Equation (21)
		Sum = 0;
	}

	return Sum;
}

void forwardProp() {
	
	DATA_TYPE Fsum = 0;
	int maxIndx = 0;
	// Propagating through network
	for (int i = 0; i < numLayers; i++) {
		// assigning node values straight from input for first layer
		if (i == 0) {
			for (int j = 0; j < NN_def[0];j++) {
				L[i].Neu[j].X = input[j];
			}
		} else if (i == numLayers - 1) {
      // softmax functionality but require normalizing performed later
			for (int j = 0; j < NN_def[i];j++) {
				y[j] = AccFunction(i,j);
				// tracking the max index
				if ( ( j > 0 ) && (abs(y[maxIndx]) < abs(y[j])) ) {
					maxIndx = j;
				}
			}
		} else {	
			// for subsequent layers, we need to perform RELU
			for (int j = 0; j < NN_def[i];j++) {
				L[i].Neu[j].X = ACT(AccFunction(i,j));				// Equation (21)	
			}	
		}
	}

  // performing exp but ensuring we dont exceed 709 or 88 in any terms 
	DATA_TYPE norm = abs(y[maxIndx]);
	if (norm > EXP_LIMIT) {
#if DEBUG
		Serial.print("Max limit exceeded for exp:");
		Serial.println(norm);
#endif
		norm = norm / EXP_LIMIT;
#if DEBUG
		Serial.print("New divising factor:");
		Serial.println(norm);
#endif
	} else {
		norm = 1.0;
	}
	for (int j = 0; j < NN_def[numLayers-1];j++) {
		int flag = 0;
		y[j] = EXP(y[j]/norm);
		Fsum += y[j];
	}

  // final normalizing for softmax
	for (int j = 0; j < NN_def[numLayers-1];j++) {
		y[j] = y[j]/Fsum;
	}
}

void backwardProp() {
	for (int i = numLayers - 1; i > 1; i--) {
    // tracing each node in the layer.
		for (int j = 0; j < NN_def[i]; j++) {
		// first checking if drivative of activation function is 0 or not! NEED TO UPGRADE TO ALLOW ACTIVATION FUNCTION OTHER THAN RELU
		L[i].Neu[j].dA = dLossCalc(i, j);

		for (int k = 0; k < NN_def[i - 1]; k++) {
			L[i].Neu[j].dW[k] = -LEARNING_RATE * L[i].Neu[j].dA * L[i - 1].Neu[k].X;
		}
		L[i].Neu[j].dB = -LEARNING_RATE * L[i].Neu[j].dA;
    }
  }
}


// function to set the input and output vectors for training or inference
void generateTrainVectors(int indx) {

	// Train Data
	for (int j = 0; j < OUT_VEC_SIZE; j++) {
		hat_y[j] = 0.0;
	}
	hat_y[ train_labels[ indxArray[indx] ] ] = 1.0;

	for (int j = 0; j < IN_VEC_SIZE; j++) {
		input[j] = cnn_train_data[ indxArray[indx] ][j];
	}

}

void shuffleIndx()
{
  for (int i = 0; i < train_data_cnt - 1; i++)
  {
    size_t j = i + rand() / (RAND_MAX / (train_data_cnt - i) + 1);
    int t = indxArray[j];
    indxArray[j] = indxArray[i];
    indxArray[i] = t;
  }
}

int calcTotalWeightsBias()
{
	int Count = 0;
	for (int i = 0; i < numLayers - 1; i++) {
		Count += NN_def[i] * NN_def[i + 1] + NN_def[i + 1];
	}

	return Count;
}

void printAccuracy()
{
  // checking accuracy if training data
  int correctCount = 0;

  for (int i = 0; i < numTrainData; i++) {
    int maxIndx = 0;
    for (int j = 0; j < IN_VEC_SIZE; j++) {
      input[j] = cnn_train_data[i][j];
    }

    forwardProp();
    for (int j = 1; j < OUT_VEC_SIZE; j++) {
      if (y[maxIndx] < y[j]) {
        maxIndx = j;
      }
    }
    if (maxIndx == train_labels[i]) {
      correctCount += 1;
    }
  }

  float Accuracy = correctCount * 1.0 / numTrainData;
  Serial.print("Training Accuracy: ");
  Serial.println(Accuracy);

  correctCount = 0;
  for (int i = 0; i < numValData; i++) {
    int maxIndx = 0;
    for (int j = 0; j < IN_VEC_SIZE; j++) {
      input[j] = cnn_validation_data[i][j];
    }

    forwardProp();
    for (int j = 1; j < OUT_VEC_SIZE; j++) {
      if (y[maxIndx] < y[j]) {
        maxIndx = j;
      }
    }
    if (maxIndx == validation_labels[i]) {
      correctCount += 1;
    }
  }

  Accuracy = correctCount * 1.0 / numValData;
  Serial.print("Validation Accuracy: ");
  Serial.println(Accuracy);

  correctCount = 0;
  for (int i = 0; i < numTestData; i++) {
    int maxIndx = 0;
    for (int j = 0; j < IN_VEC_SIZE; j++) {
      input[j] = cnn_test_data[i][j];
    }

    forwardProp();
    for (int j = 1; j < OUT_VEC_SIZE; j++) {
      if (y[maxIndx] < y[j]) {
        maxIndx = j;
      }
    }
    if (maxIndx == test_labels[i]) {
      correctCount += 1;
    }
  }

  Accuracy = correctCount * 1.0 / numTestData;
  Serial.print("Test Accuracy: ");
  Serial.println(Accuracy);
}


#define PACK 0
#define UNPACK 1
#define AVERAGE 2
// 0 -> pack vector for creating vector based on local NN for bluetooth transmission
// 1 -> unpack vector for updating weights on local NN after receiving vector via bluetooth transmission
// 2 -> average between values in pointer and location network values, and update both local NN and pointer value
void packUnpackVector(int Type)
{
  int ptrCount = 0;
  if (Type == PACK) {
    // Propagating through network, we store all weights first and then bias.
    // we start with left most layer, and top most node or lowest to highest index
    for (int i = 1; i < numLayers; i++) {
      for (int j = 0; j < NN_def[i]; j++) {
        for (int k = 0; k < L[i].Neu[j].numInput; k++) {
          WeightBiasPtr[ptrCount] = L[i].Neu[j].W[k];
          ptrCount += 1;
        }
        WeightBiasPtr[ptrCount] = L[i].Neu[j].B;
        ptrCount += 1;
      }
    }

    //Serial.print("Total count when packing:");
    //Serial.println(ptrCount);

  } else if (Type == UNPACK) {
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
  } else if (Type == AVERAGE) {
    // Propagating through network, we store all weights first and then bias.
    // we start with left most layer, and top most node or lowest to highest index
    for (int i = 1; i < numLayers; i++) {
      for (int j = 0; j < NN_def[i]; j++) {
        for (int k = 0; k < L[i].Neu[j].numInput; k++) {
          L[i].Neu[j].W[k] = (WeightBiasPtr[ptrCount] + L[i].Neu[j].W[k] ) / 2;
          WeightBiasPtr[ptrCount] = L[i].Neu[j].W[k];
          ptrCount += 1;
        }
        L[i].Neu[j].B = (WeightBiasPtr[ptrCount] + L[i].Neu[j].B ) / 2;
        WeightBiasPtr[ptrCount] = L[i].Neu[j].B;
        ptrCount += 1;
      }
    }
  }
}

// Called from main in setup-function
void setupNN(DATA_TYPE* wbptr) {
  WeightBiasPtr = wbptr;
  createNetwork();
}
