#include <iostream>

#include <NeuralNetworks/TrainingData.h>
#include <Types.h>

#include <NeuralNetworks/Network.h>

#include <map>

static constexpr MathDomain md = MathDomain::Double;
std::map<MathDomain, std::string> extension = { {MathDomain::Float, "Single"},
												{MathDomain::Double, "Double"}};

template<MathDomain T>
nn::TrainingData<T> GetData(const std::string& fileType, const size_t nRowsInput, const size_t nRowsOutput, const size_t nCols)
{
	const std::string path = getenv("DATA_PATH");
	
	auto input = cl::MatrixFromBinaryFile<MemorySpace::Device, T>(path + "/Data/" + fileType + "Input" + extension[md] + ".npy", true);
	if (input.nRows() != nRowsInput) std::abort();
	if (input.nCols() != nCols) std::abort();
	
	auto output = cl::MatrixFromBinaryFile<MemorySpace::Device, T>(path + "/Data/" + fileType + "Output" + extension[md] + ".npy", true);
	if (output.nRows() != nRowsOutput) std::abort();
	if (output.nCols() != nCols) std::abort();
	
	return nn::TrainingData<T>(std::move(input), std::move(output));
}

int main()
{
	auto trainingData = GetData<md>("Training", 784, 10, 50000);
	auto validationData = GetData<md>("Validation", 784, 10, 10000);
	auto testData = GetData<md>("Test", 784, 10, 10000);
	
	nn::Vector<MathDomain::Int> cache1(10000u);
	nn::Vector<MathDomain::Int> cache2(10000u);
	nn::Vector<MathDomain::Int> cache3(10000u);
	std::function<void(nn::Matrix<md>&, const nn::Matrix<md>&)> evaluator = [&cache1, &cache2, &cache3](nn::Matrix<md>& modelOutput, const nn::Matrix<md>& expectedOutput)
	{
		assert(modelOutput.nCols() == cache1.size());
		modelOutput.ColumnWiseArgAbsMaximum(cache1);
		
		assert(expectedOutput.nCols() == cache2.size());
		expectedOutput.ColumnWiseArgAbsMaximum(cache2);
		
		int score = cache1.CountEquals(cache2, cache3.GetBuffer());
		
		std::cout << "\t***\tScore = " << score << " [" << modelOutput.nCols() << "] ***" << std::endl;
	};
	
	nn::NetworkTrainingData<md> data(trainingData, testData, validationData, evaluator);
	data.debugLevel = 2;
	data.epochCalculation = 1;
	
	data.hyperParameters.nEpochs = 30;
	data.hyperParameters.miniBacthSize = 10;
	data.hyperParameters.learningRate = 3.0;
	
	auto networkTopology = std::vector<size_t>{{ 784, 30, 10 }};
	nn::Network<md> network(networkTopology);
	network.Train(data);
	return 0;
}