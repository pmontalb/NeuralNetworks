#include <iostream>

#include <NeuralNetworks/TrainingData.h>
#include <NeuralNetworks/Network.h>

#include <NeuralNetworks/Optimizers/All.h>
#include <NeuralNetworks/Layers/Initializers/All.h>
#include <NeuralNetworks/CostFunctions/All.h>
#include <NeuralNetworks/Layers/All.h>
#include <NeuralNetworks/Activations/All.h>

#include <NeuralNetworks/Activations/ActivationFunctionFactory.h>

#include <map>

static constexpr MathDomain md = MathDomain::Float;

template<MathDomain T>
nn::TrainingData<T> GetData(const std::string& fileType, const size_t nRowsInput, const size_t nRowsOutput, const size_t nCols)
{
	const std::map<MathDomain, std::string> extension = { {MathDomain::Float, "Single"},
													      {MathDomain::Double, "Double"}};
	
	const std::string path = getenv("DATA_PATH");
	
	auto input = cl::ColumnWiseMatrix<MemorySpace::Device, T>::MatrixFromBinaryFile(path + "/Data/" + fileType + "Input" + extension.at(md) + ".npy", false, true);
	if (input.nRows() != nRowsInput) std::abort();
	if (input.nCols() != nCols) std::abort();
	
	auto output = cl::ColumnWiseMatrix<MemorySpace::Device, T>::MatrixFromBinaryFile(path + "/Data/" + fileType + "Output" + extension.at(md) + ".npy", false, true);
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
		MemoryBuffer cache4;
		MemoryBuffer cache5;

		explicit Cache(const unsigned N)
			: cache1(N), cache2(N), cache3(N)
		{
			cache4 = MemoryBuffer(0, 1, MemorySpace::Device, MathDomain::Int);
			dm::detail::Alloc(cache4);

			dm::detail::DetermineSumCache(cache5, cache3.GetBuffer(), cache4);
		}
		Cache(Cache&& rhs) noexcept
			: cache1(std::move(rhs.cache1)),
			  cache2(std::move(rhs.cache2)),
			  cache3(std::move(rhs.cache3)),
			  cache4(std::move(rhs.cache4)),
			  cache5(std::move(rhs.cache5))
		{
			rhs.cache4.pointer = 0;
			rhs.cache5.pointer = 0;
		}

		~Cache()
		{
			if (cache4.pointer)
				dm::detail::Free(cache4);
			if (cache5.pointer)
				dm::detail::Free(cache5);
		}
		Cache(const Cache&) = delete;
		Cache& operator=(const Cache&) = delete;
		Cache& operator=(Cache&&) = delete;
	};
	std::map<unsigned, Cache> caches;
	std::function<double(nn::Matrix<md>&, const nn::Matrix<md>&)> evaluator = [&caches](nn::Matrix<md>& modelOutput, const nn::Matrix<md>& expectedOutput)
	{
		auto iter = caches.find(modelOutput.nCols());
		if (iter == caches.end())
			iter = caches.emplace(std::piecewise_construct,
			                      std::forward_as_tuple(modelOutput.nCols()),
			                      std::forward_as_tuple(Cache(modelOutput.nCols()))).first;

		assert(modelOutput.nCols() == iter->second.cache1.size());
		modelOutput.ColumnWiseArgAbsMaximum(iter->second.cache1);

		assert(expectedOutput.nCols() == iter->second.cache2.size());
		expectedOutput.ColumnWiseArgAbsMaximum(iter->second.cache2);

		int score = iter->second.cache1.CountEquals(iter->second.cache2, iter->second.cache3.GetBuffer(), iter->second.cache5, iter->second.cache4);

		std::cout << "\t***\tScore = " << score << " [" << modelOutput.nCols() << "] = " << static_cast<double>(100 * score) / modelOutput.nCols() << "% ***" << std::endl;

		return static_cast<double>(score);
	};

	nn::NetworkTrainingData<md> data(trainingData, testData, validationData, evaluator);
	data.debugLevel = 1;
	data.epochCalculationAccuracyTestData = 1;
	data.nMaxEpochsWithNoScoreImprovements = 500;

	data.hyperParameters.nEpochs = 1000;
	data.hyperParameters.miniBatchSize = 10;
	data.hyperParameters.learningRate = 0.1;
	data.hyperParameters.lambda = 5.0;

	std::vector<std::unique_ptr<nn::ILayer<md>>> layers;
//	networkTopology.emplace_back(std::make_unique<nn::DenseLayer<md>>(784, 100, std::make_unique<nn::SigmoidActivationFunction<md>>(), nn::SmallVarianceRandomBiasWeightInitializer<md>()));
	layers.emplace_back(std::make_unique<nn::DenseLayer<md>>(784, 100, std::make_unique<nn::SigmoidActivationFunction<md>>(), nn::SmallVarianceRandomBiasWeightInitializer<md>()));
//	layers.emplace_back(std::make_unique<nn::DenseLayer<md>>(100, 10,  std::make_unique<nn::SigmoidActivationFunction<md>>(), nn::SmallVarianceRandomBiasWeightInitializer<md>()));
//	layers.emplace_back(std::make_unique<nn::SoftMaxLayer<md>>(100, 10,  std::make_unique<nn::SoftMaxActivationFunction<md>>(), nn::SmallVarianceRandomBiasWeightInitializer<md>()));
	layers.emplace_back(std::make_unique<nn::SoftMaxLayer<md>>(100, 10, std::make_unique<nn::SoftMaxActivationFunction<md>>(), nn::ZeroBiasWeightInitializer<md>()));
	nn::Network<md> network((nn::NetworkTopology<md>(std::move(layers))));

	nn::BatchedSgd<md> optimizer(network.GetTopology(), data.hyperParameters.miniBatchSize, std::make_unique<nn::LogLikelihoodCostFunction<md>>(), std::make_unique<nn::RandomShuffler<md>>());
//	nn::BatchedSgd<md> optimizer(network.GetTopology(), data.hyperParameters.miniBatchSize, std::make_unique<nn::LogLikelihoodCostFunction<md>>(), std::make_unique<nn::IdentityShuffler<md>>());
//	nn::BatchedSgd<md> optimizer(network.GetTopology(), data.hyperParameters.miniBatchSize, std::make_unique<nn::CrossEntropyCostFunction<md>>(), std::make_unique<nn::IdentityShuffler<md>>());
//	nn::BatchedSgd<md> optimizer(network.GetTopology(), data.hyperParameters.miniBatchSize, std::make_unique<nn::CrossEntropyCostFunction<md>>(), std::make_unique<nn::RandomShuffler<md>>());
	network.Train(optimizer, data);
	return 0;
}
