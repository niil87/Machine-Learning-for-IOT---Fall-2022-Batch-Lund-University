// Code developed by Nikhil Challa as part of ML in IOT Course - year : 2022
// Team members : Simon Erlandsson

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define fRAND ( rand()*1.0/RAND_MAX-0.5 )*2   // random number generator between -1 and +1 
#define ACT(a) MAX(a,0)    // RELU(a)

// NN parameters
#define LEARNING_RATE 0.01
#define EPOCH 25
#define IN_VEC_SIZE first_layer_input_cnt
#define OUT_VEC_SIZE classes_cnt

// Defining the network structure. 
// Input size is equal to first layer size. Same for output except we need to do one additional function
// Final Output we will need to handle differently depending on case, in ours its softmax
static const int NN_def[] = {IN_VEC_SIZE,  20, OUT_VEC_SIZE};
size_t numLayers = sizeof(NN_def)/sizeof(NN_def[0]);
// size of the input to NN

// dummy input for testing
float input[IN_VEC_SIZE];

// dummy output for testing
float hat_y[OUT_VEC_SIZE];    // target output
float y[OUT_VEC_SIZE];        // output after forward propagation


// creating array index to randomnize order of training data
int indxArray[train_data_cnt];

// Convention:
// Aggregated results from all weights to node is A
// Application of activation function to A is X
// Weights will be represented by W and is weights entering the node
// Bias will be represented by B
typedef struct neuron_t {
	int numInput;
	float* W;     
	float B;
	float X;
	
	// For back propagation, convention, dA means dL/dA or partial derivative of Loss over Accumulative output
	float* dW;
	float dA;
	float dB;
	
} neuron;

typedef struct layer_t {
	int numNeuron;
	neuron* Neu;
} layer;

// initializing the layer as global parameter 
layer* L = NULL;


float AccFunction (int layerIndx, int nodeIndx) {
	float A = 0;

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
	N1.W = (float*)malloc(numInput*sizeof(float));
	N1.dW = (float*)malloc(numInput*sizeof(float));
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
float dLossCalc( int layerIndx, int nodeIndx) {
	
	float Sum = 0;
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
	
	float Fsum = 0;
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
			for (int j = 0; j < NN_def[i];j++) {
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
