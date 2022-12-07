#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

#include <ArduinoBLE.h>

//#include <EloquentTinyML.h>
//#include <eloquent_tinyml/tensorflow.h>

#include "cnn_data.h" // network weights

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define fRAND ( rand()*1.0/RAND_MAX-0.5 )*2   // random number generator between -1 and +1   // this approach is failing sometimes.. need to understand why -ve weights are not working

#define ACT(a) MAX(a,0)    // RELU(a)

	
// NN parameters
#define LEARNING_RATE 0.1
#define EPOCH 1000
#define IN_VEC_SIZE 75
#define OUT_VEC_SIZE 3

// Defining the network structure. 
// Input size is equal to first layer size. Same for output except we need to do one additional function
// Final Output we will need to handle differently depending on case, in ours its softmax
static const int NN_def[] = {IN_VEC_SIZE,  75, 35, 15, OUT_VEC_SIZE};
size_t numLayers = sizeof(NN_def)/sizeof(NN_def[0]);
// size of the input to NN


// size of different vectors
size_t numValData = sizeof(validation_labels) / sizeof(int);
size_t numTrainData = sizeof(train_labels) / sizeof(int);

// creating array index to randomnize order of training data
//int indxArray[numTrainData];
int indxArray[73];
int shuffle = 0;


// dummy input for testing
double input[IN_VEC_SIZE];

// dummy output for testing
double hat_y[OUT_VEC_SIZE];    // target output
double y[OUT_VEC_SIZE];        // output after forward propagation


// Convention:
// Aggregated results from all weights to node is A
// Application of activation function to A is X
// Weights will be represented by W and is weights entering the node
// Bias will be represented by B
typedef struct neuron_t {
	int numInput;
	double* W;     
	double B;
	double X;
	
	// For back propagation, convention, dA means dL/dA or partial derivative of Loss over Accumulative output
	double* dW;
	double dA;
	double dB;
	
} neuron;

typedef struct layer_t {
	int numNeuron;
	neuron* Neu;
} layer;
	
// initializing the layer as global parameter 
layer* L = NULL;


double AccFunction (int layerIndx, int nodeIndx) {
	double A = 0;

	for (int k = 0; k < NN_def[layerIndx - 1]; k++) {

		// updating weights/bais and resetting gradient value if non-zero
		if (L[layerIndx].Neu[nodeIndx].dW[k] != 0.0 ) {
			L[layerIndx].Neu[nodeIndx].W[k] += L[layerIndx].Neu[nodeIndx].dW[k];
			L[layerIndx].Neu[nodeIndx].dW[k] = 0.0;	
		}
		
		A += L[layerIndx].Neu[nodeIndx].W[k] * L[layerIndx-1].Neu[k].X;	
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
	N1.W = (double*)malloc(numInput*sizeof(double));
	N1.dW = (double*)malloc(numInput*sizeof(double));
	// initializing values of W to rand and dW to 0
	int Sum = 0;
	for (int i=0; i < numInput; i++) {
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
	L1.Neu = (neuron*)malloc(numNeuron*sizeof(neuron));
	return L1;
	
}

void createNetwork() {
	
	L = (layer*)malloc(numLayers*sizeof(layer));
	
	// First layer has no input weights
	L[0] = createLayer(NN_def[0]);
	
	for (int i = 1; i < numLayers; i++) {
			L[i] = createLayer(NN_def[i]);
		for (int j = 0; j < NN_def[i]; j++) {
			L[i].Neu[j] = createNeuron(NN_def[i-1]);
		}
	}
	
}


// this function is to calculate dA
double dLossCalc( int layerIndx, int nodeIndx) {
	
	double Sum = 0;
	int outputSize = NN_def[numLayers-1];
	// for the last layer, we use complex computation
	if (layerIndx == numLayers - 1) {

		Sum = y[nodeIndx] - hat_y[nodeIndx];

	// for all except last layer, we use simple aggregate of dA
	} else if (AccFunction(layerIndx,nodeIndx) > 0)  {

		for (int i = 0; i < NN_def[layerIndx + 1]; i++) {
			Sum += L[layerIndx + 1].Neu[i].dA * L[layerIndx + 1].Neu[i].W[nodeIndx];
		}
		
	} else {
		Sum = 0;
	}
	
	return Sum;
}

void forwardProp() {
	
	double Fsum = 0;
	// Propagating through network
	for (int i = 0; i < numLayers; i++) {
		// assigning node values straight from input for first layer
		if (i == 0) {
			for (int j = 0; j < NN_def[0];j++) {
				L[i].Neu[j].X = input[j];
			}
		} else if (i == numLayers - 1) {

			for (int j = 0; j < NN_def[i];j++) {
				y[j] = exp(AccFunction(i,j));
				Fsum += y[j];
			}
		} else {
			// for subsequent layers, we need to perform RELU
			// printf("Layer:%d\n",i);
			for (int j = 0; j < NN_def[i];j++) {
				// printf("Accumulation value:%f\n",AccFunction(i,j));
				L[i].Neu[j].X = ACT(AccFunction(i,j));
			}	
		}
	}

	for (int j = 0; j < NN_def[numLayers-1];j++) {
		y[j] = y[j]/Fsum;
	}
	
}

void backwardProp() { 
	
	for (int i = numLayers-1; i > 1; i--) {
		// tracing each node in the layer.
		for (int j = 0;j < NN_def[i];j++) {
			// first checking if drivative of activation function is 0 or not! NEED TO UPGRADE TO ALLOW ACTIVATION FUNCTION OTHER THAN RELU
			L[i].Neu[j].dA = dLossCalc(i,j);

			for (int k = 0; k < NN_def[i - 1]; k++) {
				L[i].Neu[j].dW[k] = -LEARNING_RATE * L[i].Neu[j].dA * L[i-1].Neu[k].X;

			}
			L[i].Neu[j].dB = -LEARNING_RATE * L[i].Neu[j].dA;

		}
	}
}

// function to set the input and output vectors for training or inference
// dataType = 0 Training data
// dataType = 1 Validation data
void generateData(int indx, int dataType) {
	
	// for every epoch, we need to perform indx randomnization for better training
	if (dataType == 0) {
		if (shuffle == 1) {
			size_t i;
			for (i = 0; i < numTrainData - 1; i++) 
			{
			  size_t j = i + rand() / (RAND_MAX / (numTrainData - i) + 1);
			  int t = indxArray[j];
			  indxArray[j] = indxArray[i];
			  indxArray[i] = t;
			}
		}
		shuffle = 0;

		// Train Data
		for (int j =0; j < OUT_VEC_SIZE; j++) {
			hat_y[j] = 0.0;
		}
		hat_y[ indxArray[indx] ] = 1.0;
		
		for (int j =0; j < IN_VEC_SIZE; j++) {
			input[j] = cnn_train_data[ indxArray[indx] ][j];
		}
	
	} else {
		
		for (int j =0; j < IN_VEC_SIZE; j++) {
			input[j] = cnn_validation_data[indx][j];
		}
	}

}



void setup() {
	// put your setup code here, to run once:
	Serial.begin(9600);
	Serial.println("Hello World!");
	
	// randomly initialize the seed based on time
	srand(time(0));
	
  for (int i = 0; i <  numTrainData; i ++ ) {
    indxArray[i] = i;
  }

	createNetwork();

	// looping to check if forward+backward converges
	for (int i = 0; i < EPOCH; i++) {
		Serial.print("Loop count:");
		Serial.print(i);
		Serial.println();
		shuffle = 1;
		
    Serial.println("#################### starting forward + Backward propagation ##################### \n");
		for (int j = 0;j < numTrainData;j++) {
			generateData(j,0);	
			forwardProp();
			backwardProp();
		}

	}
	
	// checking accuracy if validation data
	int ErrorCount = 0;
	int maxIndx = 0;
	for (int i = 0; i < numValData; i++) {
		generateData(i,1);
		forwardProp();
		for (int j = 1; j < OUT_VEC_SIZE;j++) {
			if (y[maxIndx] < y[j]) {
				maxIndx = j;
			}
		}
		if (maxIndx != validation_labels[i]) {
			ErrorCount+=1;
		}
	}
	
	float Accuracy = ErrorCount/numValData;
	Serial.println("## Accuracy:");
	Serial.print(Accuracy);

}

void loop() {
  // put your main code here, to run repeatedly:

}
