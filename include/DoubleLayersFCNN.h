#pragma once
class DoubleLayersFCNN {
public:
	DoubleLayersFCNN(int _numberInputNeurons, int _numberHiddenNeurons, int _numberOutputNeurons);
	~DoubleLayersFCNN();

	void train(double **data, double *label, int sampleSize, int numberEpochs, double learningRate, double errorCrossEntropy);
	double precision(double **data, double *label, int sampleSize);
	void cleanNetWeights();
private:
	int inputSize;
	int hiddenSize;
	int outputSize;

	double *inputLayer;
	double *hiddenLayer;
	double *outputLayerFact;
	double *outputLayerExpected;
	double *weightsLayer1;
	double *weightsLayer2;

	void initWeights();
	void calculateOutputLayer();
	void calculateHiddenLayer();
	void calculateGradients(double *gradientWeightsLayer1, double *gradientWeightsLayer2);
	void correctWeights(double *gradientWeightsLayer1, double *gradientWeightsLayer2, double learningRate);

	void mix(int *order, int size);
	void setInputOutputExpectedLayers(double * input, double lable);
	int getIndexOfMax(double * arr, int size);

	double crossEntropy(double **data, double *label, int sampleSize);
	double dtanh(double valueTanh);
	double* softmax(double *g, int numberNeurons);
};