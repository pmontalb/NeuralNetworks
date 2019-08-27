#include <iostream>

#include <NeuralNetworks/TrainingData.h>
#include <Types.h>

#include <NeuralNetworks/Network.h>
#include <NeuralNetworks/Shufflers/IdentityShuffler.h>
#include <NeuralNetworks/Initializers/ZeroBiasWeightInitializer.h>

#include <map>

static constexpr MathDomain md = MathDomain::Double;

template<MathDomain T>
nn::TrainingData<T> GetData(const std::string& fileType, const size_t nRowsInput, const size_t nRowsOutput, const size_t nCols)
{
	const std::map<MathDomain, std::string> extension = { {MathDomain::Float, "Single"},
													      {MathDomain::Double, "Double"}};
	
	const std::string path = getenv("DATA_PATH");
	
	auto input = cl::MatrixFromBinaryFile<MemorySpace::Device, T>(path + "/Data/" + fileType + "Input" + extension.at(md) + ".npy", true);
	if (input.nRows() != nRowsInput) std::abort();
	if (input.nCols() != nCols) std::abort();
	
	auto output = cl::MatrixFromBinaryFile<MemorySpace::Device, T>(path + "/Data/" + fileType + "Output" + extension.at(md) + ".npy", true);
	if (output.nRows() != nRowsOutput) std::abort();
	if (output.nCols() != nCols) std::abort();
	
	return nn::TrainingData<T>(std::move(input), std::move(output));
}

int main()
{
	auto trainingData = GetData<md>("Training", 784, 10, 50000);
	auto validationData = GetData<md>("Validation", 784, 10, 10000);
	auto testData = GetData<md>("Test", 784, 10, 10000);

	struct Cache
	{
		nn::Vector<MathDomain::Int> cache1;
		nn::Vector<MathDomain::Int> cache2;
		nn::Vector<MathDomain::Int> cache3;
		
		explicit Cache(const unsigned N)
			: cache1(N), cache2(N), cache3(N)
		{
		}
	};
	std::map<unsigned, Cache> caches;
	std::function<double(nn::Matrix<md>&, const nn::Matrix<md>&)> evaluator = [&caches](nn::Matrix<md>& modelOutput, const nn::Matrix<md>& expectedOutput)
	{
		const auto iter = caches.emplace(modelOutput.nCols(), Cache(modelOutput.nCols())).first;
		
		assert(modelOutput.nCols() == iter->second.cache1.size());
		modelOutput.ColumnWiseArgAbsMaximum(iter->second.cache1);
		
		assert(expectedOutput.nCols() == iter->second.cache2.size());
		expectedOutput.ColumnWiseArgAbsMaximum(iter->second.cache2);
		
		int score = iter->second.cache1.CountEquals(iter->second.cache2, iter->second.cache3.GetBuffer());
		
		std::cout << "\t***\tScore = " << score << " [" << modelOutput.nCols() << "] ***" << std::endl;
		
		return static_cast<double>(score);
	};
	
	nn::NetworkTrainingData<md> data(trainingData, testData, validationData, evaluator);
	data.debugLevel = 2;
	data.epochCalculationTestData = 0;
	data.epochCalculationValidationData = 0;
	data.epochCalculationTrainingData = 1;
	
	data.hyperParameters.nEpochs = 30;
	data.hyperParameters.miniBacthSize = 10;
	data.hyperParameters.learningRate = 0.1;
	data.hyperParameters.lambda = 5.0;
	
	auto networkTopology = std::vector<size_t>{{ 784, 30, 10 }};
	nn::Network<md> network(networkTopology,
			                nn::ZeroBiasWeightInitializer<md>(),
			                std::make_unique<nn::CrossEntropyCostFunction<md>>(),
			                std::make_unique<nn::IdentityShuffler<md>>());
	network.Train(data);
	return 0;
}
