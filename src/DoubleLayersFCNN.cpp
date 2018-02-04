#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <math.h>

#include "DoubleLayersFCNN.h"

DoubleLayersFCNN::DoubleLayersFCNN(int _numberInputNeurons, int _numberHiddenNeurons, int _numberOutputNeurons) {
	inputSize = _numberInputNeurons;
	hiddenSize = _numberHiddenNeurons;
	outputSize = _numberOutputNeurons;

	inputLayer = new double[inputSize];
	hiddenLayer = new double[hiddenSize];
	outputLayerFact = new double[outputSize];
	outputLayerExpected = new double[outputSize];

	weightsLayer1 = new double[inputSize*hiddenSize];
	weightsLayer2 = new double[hiddenSize*outputSize];
	initWeights();
}

DoubleLayersFCNN::~DoubleLayersFCNN() {
	delete[] inputLayer;
	delete[] outputLayerExpected;
	delete[] hiddenLayer;
	delete[] outputLayerFact;
	delete[] weightsLayer1;
	delete[] weightsLayer2;
}

void DoubleLayersFCNN::initWeights() {
	srand(time(NULL));
	for (int i = 0; i < inputSize*hiddenSize; i++)
		weightsLayer1[i] = (double(rand()) / (double)RAND_MAX) / 100.0;

	for (int i = 0; i < hiddenSize*outputSize; i++)
		weightsLayer2[i] = (double(rand()) / (double)RAND_MAX) / 100.0;
}

void DoubleLayersFCNN::calculateHiddenLayer() {
	double *f = new double[hiddenSize];

	for (int s = 0; s < hiddenSize; s++) {
		f[s] = 0;
		for (int i = 0; i < inputSize; i++) {
			f[s] += weightsLayer1[i*hiddenSize + s] * inputLayer[i];
		}
		hiddenLayer[s] = tanh(f[s]);
	}
	hiddenLayer[0] = 1;

	delete[] f;
}

void DoubleLayersFCNN::calculateOutputLayer() {
	double *g = new double[outputSize];

	calculateHiddenLayer();

	for (int j = 0; j < outputSize; j++) {
		g[j] = 0;
		for (int s = 0; s < hiddenSize; s++) {
			g[j] += weightsLayer2[s*outputSize + j] * hiddenLayer[s];
		}
	}

	outputLayerFact = softmax(g, outputSize);

	delete[] g;
}

void DoubleLayersFCNN::calculateGradients(double *gradient1, double *gradient2) {
	double *sigmaLayer2 = new double[outputSize];
	double *summa = new double[hiddenSize];
	double *dActFuncHiddenLayer = new double[hiddenSize];

	for (int s = 0; s < hiddenSize; s++) {
		for (int j = 0; j < outputSize; j++) {
			sigmaLayer2[j] = outputLayerFact[j] - outputLayerExpected[j];
			gradient2[s*outputSize + j] = sigmaLayer2[j] * hiddenLayer[s];
		}
	}

	for (int s = 0; s < hiddenSize; s++) {
		dActFuncHiddenLayer[s] = dtanh(hiddenLayer[s]);
	}

	for (int s = 0; s < hiddenSize; s++) {
		summa[s] = 0;
		for (int j = 0; j < outputSize; j++) {
			summa[s] += sigmaLayer2[j] * weightsLayer2[s*outputSize + j];
		}
	}

	for (int i = 0; i < inputSize; i++) {
		for (int s = 0; s < hiddenSize; s++) {
			gradient1[i*hiddenSize + s] = dActFuncHiddenLayer[s] * summa[s] * inputLayer[i];
		}
	}

	delete[] sigmaLayer2;
	delete[] summa;
	delete[] dActFuncHiddenLayer;
}

void DoubleLayersFCNN::correctWeights(double *gradientWeightsLayer1, double *gradientWeightsLayer2, double learningRate)
{
	for (int i = 0; i < inputSize*hiddenSize; i++)
		weightsLayer1[i] -= learningRate * gradientWeightsLayer1[i];

	for (int s = 0; s < hiddenSize*outputSize; s++)
		weightsLayer2[s] -= learningRate * gradientWeightsLayer2[s];
}

double DoubleLayersFCNN::crossEntropy(double **trainData, double *trainLabel, int sampleSize) {
	double crossEntropy = 0;
	for (int image = 0; image < sampleSize; image++) {
		setInputOutputExpectedLayers(trainData[image], trainLabel[image]);
		calculateOutputLayer();
		for (int j = 0; j < outputSize; j++) {
			crossEntropy += outputLayerExpected[j] * log(outputLayerFact[j]);
		}
	}
	return -1 * crossEntropy / sampleSize;
}

double DoubleLayersFCNN::precision(double **data, double *label, int sampleSize)
{
	int truePrediction = 0;
	for (int item = 0; item < sampleSize; item++) {
		setInputOutputExpectedLayers(data[item], label[item]);
		calculateOutputLayer();
		if (fabs(outputLayerExpected[getIndexOfMax(outputLayerFact, outputSize)] - 1.0) < 0.0000000001)
			truePrediction++;
	}
	return (double)truePrediction / (double)(sampleSize);
}

void DoubleLayersFCNN::cleanNetWeights()
{
	initWeights();
}

void DoubleLayersFCNN::train(double **data, double *label, int sampleSize, int numberEpochs, double learningRate, double errorCrossEntropy) {
	double *gradientWeightsLayer1 = new double[inputSize*hiddenSize];
	double *gradientWeightsLayer2 = new double[hiddenSize*outputSize];
	int *order = new int[sampleSize];

	for (int epoch = 0; epoch < numberEpochs; epoch++) {
		mix(order, sampleSize);
		for (int image = 0; image < sampleSize; image++) {
			setInputOutputExpectedLayers(data[order[image]], label[order[image]]);
			calculateOutputLayer();
			calculateGradients(gradientWeightsLayer1, gradientWeightsLayer2);
			correctWeights(gradientWeightsLayer1, gradientWeightsLayer2, learningRate);
		}

		double currentCrossEntropy = crossEntropy(data, label, sampleSize);
		printf("-- Epoch = %d, CrossEntropy = %f \n", epoch, currentCrossEntropy);
		if (currentCrossEntropy < errorCrossEntropy) {
			break;
		}
	}

	delete[] order;
	delete[] gradientWeightsLayer1;
	delete[] gradientWeightsLayer2;
}

void DoubleLayersFCNN::mix(int *arr, int size) {
	int randomNumber, tmp;
	for (int i = 0; i < size; i++) {
		arr[i] = i;
	}

	for (int i = 0; i < size; i++) {
		randomNumber = i + rand() % (size - i);
		tmp = arr[i];
		arr[i] = arr[randomNumber];
		arr[randomNumber] = tmp;
	}
}

double DoubleLayersFCNN::dtanh(double x)
{
	return (1 - x) * (1 + x);
}

double * DoubleLayersFCNN::softmax(double * x, int size)
{
	double* res = new double[size];
	double sumExp = 0;

	for (int i = 0; i < size; i++) {
		sumExp += exp(x[i]);
	}

	for (int i = 0; i < size; i++) {
		res[i] = exp(x[i]) / sumExp;
	}

	return res;
}

void DoubleLayersFCNN::setInputOutputExpectedLayers(double * input, double label)
{
	for (int i = 0; i < inputSize; i++) {
		inputLayer[i] = input[i];
	}
	for (int i = 0; i < outputSize; i++) {
		outputLayerExpected[i] = 0.0;
	}
	outputLayerExpected[(int)label] = 1.0;
}

int DoubleLayersFCNN::getIndexOfMax(double * arr, int size)
{
	int maxIndex = 0;
	for (int i = 0; i < size; i++) {
		if (arr[i] > arr[maxIndex]) {
			maxIndex = i;
		}
	}
	return maxIndex;
}
