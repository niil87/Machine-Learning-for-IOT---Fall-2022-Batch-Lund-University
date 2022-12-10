#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>


#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define fRAND ( rand()*1.0/RAND_MAX-0.5 )*2   // random number generator between -1 and +1   
#define ACT(a) MAX(a,0)    // RELU(a)

	
// this is to set the precision for weight/bias in NN
#define DATA_TYPE_DOUBLE  // other value is DATA_TYPE_FLOAT

#ifdef DATA_TYPE_FLOAT 
  #define DATA_TYPE float
  #define EXP_LIMIT 78.0  // limit 88.xx but we need to factor in accumulation for softmax
  #define EXP(a) expl(a)
#else
  #define DATA_TYPE double
  #define EXP_LIMIT 699.0 // limit is 709.xx but we need to factor in accumulation for softmax
  #define EXP(a) exp(a)
#endif


// NN parameters
#define LEARNING_RATE 0.1
#define EPOCH 1000
#define IN_VEC_SIZE 20
#define OUT_VEC_SIZE 20

// Defining the network structure. 
// Input size is equal to first layer size. Same for output except we need to do one additional function
// Final Output we will need to handle differently depending on case, in ours its softmax
static const int NN_def[] = {IN_VEC_SIZE, 10,10, OUT_VEC_SIZE};
size_t numLayers = sizeof(NN_def)/sizeof(NN_def[0]);
// size of the input to NN

// dummy input for testing
DATA_TYPE input[IN_VEC_SIZE];

// dummy output for testing
DATA_TYPE hat_y[OUT_VEC_SIZE];    // target output
DATA_TYPE y[OUT_VEC_SIZE];        // output after forward propagation


// Convention:
// Aggregated results from all weights to node is A
// Application of activation function to A is X
// Weights will be represented by W and is weights entering the node
// Bias will be represented by B
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


DATA_TYPE AccFunction (int layerIndx, int nodeIndx) {
	DATA_TYPE A = 0;

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
	N1.W = (DATA_TYPE*)malloc(numInput*sizeof(DATA_TYPE));
	N1.dW = (DATA_TYPE*)malloc(numInput*sizeof(DATA_TYPE));
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
DATA_TYPE dLossCalc( int layerIndx, int nodeIndx) {
	
	DATA_TYPE Sum = 0;
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
				L[i].Neu[j].X = ACT(AccFunction(i,j));
			}	
		}
	}

  // performing exp but ensuring we dont exceed 709 or 88 in any terms 
  DATA_TYPE norm = abs(y[maxIndx]);
  if (norm > EXP_LIMIT) {
    Serial.print("Max limit exceeded for exp:");
    Serial.println(norm);
    norm = norm / EXP_LIMIT;
    Serial.print("New divising factor:");
    Serial.println(norm);
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

// REMOVE FOR DEPLOYMENT
void generateData() {
	int indx = rand()*1.0/RAND_MAX*9;

	for (int j =0; j < OUT_VEC_SIZE; j++) {

		input[j] = 0.0;
		hat_y[j] = 0.0;
	}
	input[indx] = 1.0;
	hat_y[indx] = 1.0;
}



void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.println("Hello World!");
	
	// randomly initialize the seed based on time
	srand(time(0));
	
	createNetwork();

	// looping to check if forward+backward converges
	for (int i = 0; i < EPOCH; i++) {
    Serial.print("Loop count:");
    Serial.print(i);
    Serial.println();
		generateData();
		
		Serial.println("#################### starting forward propagation ##################### \n");
		forwardProp();
		
    if (i > EPOCH - 20) {
      
      for (int j = 0; j < NN_def[numLayers-1];j++) {
        Serial.print(y[j]);
        Serial.println();
      }
      
      Serial.println();
      
      for (int j = 0; j < NN_def[numLayers-1];j++) {
        Serial.print(input[j]);
        Serial.println();
      }
    }
		
		Serial.println("#################### starting backward propagation ###################### \n\n\n");
		backwardProp();

	}


}

void loop() {
  // put your main code here, to run repeatedly:

}
