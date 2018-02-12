#include "DoubleLayersFCNN.h"
#include "ReaderMNIST.h"

struct NetSettings
{
	char *fnameTrainData;
	char *fnameTrainLabels;
	char *fnameTestData;
	char *fnameTestLabels;
	double learningRate;
	double errorCrossEntropy;
	int numberEpochs;
	int hiddenSize;

	NetSettings()
	{
		fnameTrainData = new char[250];
		fnameTrainLabels = new char[250];
		fnameTestData = new char[250];
		fnameTestLabels = new char[250];
	}
};

NetSettings getSettings(char* fname)
{
	NetSettings res;
	FILE * settings = fopen(fname, "r");
	if (!feof(settings))
	{
		char* s = new char[250];
		fscanf(settings, "%s", res.fnameTrainData);
		fscanf(settings, "%s", res.fnameTrainLabels);
		fscanf(settings, "%s", res.fnameTestData);
		fscanf(settings, "%s", res.fnameTestLabels);
		fscanf(settings, "%lf%s", &res.learningRate, s);
		fscanf(settings, "%lf%s", &res.errorCrossEntropy, s);
		fscanf(settings, "%i%s", &res.hiddenSize, s);
		fscanf(settings, "%i%s", &res.numberEpochs, s);
	}
	fclose(settings);
	return res;
}

int main(int argc, char* argv[])
{
	printf("NetSetting");
	NetSettings settings = getSettings("config/NetSettings.txt");
	int width = 28, height = 28;
	int numberTrainImage = 60000;
	int numberTestImage = 10000;
	int numberInput = width * height + 1;
	int numberOutput = 10;
	DoubleLayersFCNN network = DoubleLayersFCNN(numberInput, settings.hiddenSize + 1, numberOutput);
	printf(" - OK\n");

	printf("Reading MNIST");
	double **trainData = new double*[numberTrainImage];
	for (int i = 0; i < numberTrainImage; i++)
		trainData[i] = new double[numberInput];

	double *trainLabel = new double[numberTrainImage];

	double **testData = new double*[numberTestImage];
	for (int i = 0; i < numberTestImage; i++)
		testData[i] = new double[numberInput];

	double *testLabel = new double[numberTestImage];

	readSetImage(settings.fnameTrainData, trainData);
	readSetLabel(settings.fnameTrainLabels, trainLabel);
	readSetImage(settings.fnameTestData, testData);
	readSetLabel(settings.fnameTestLabels, testLabel);
	printf(" - OK\n");

	printf("Training ...\n");
	network.train(trainData, trainLabel, numberTrainImage, settings.numberEpochs, settings.learningRate, settings.errorCrossEntropy);

	printf("Precision ...\n");
	double precision = network.precision(trainData, trainLabel, numberTrainImage);
	printf("-- Train = %f \n", precision);
	precision = network.precision(testData, testLabel, numberTestImage);
	printf("-- Test = %f \n", precision);

	system("pause");

	for (int i = 0; i < numberTrainImage; i++)
		delete[] trainData[i];
	delete[] trainData;

	for (int i = 0; i < numberTestImage; i++)
		delete[] testData[i];
	delete[] testData;

	delete[] trainLabel;
	delete[] testLabel;
	return 0;
}