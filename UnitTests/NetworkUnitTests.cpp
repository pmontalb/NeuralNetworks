
#include <gtest/gtest.h>

#include <Network.h>
#include <NeuralNetworks/Initializers/RandomBiasWeightInitializer.h>
#include <NeuralNetworks/Initializers/SmallVarianceRandomBiasWeightInitializer.h>
#include <NeuralNetworks/CostFunctions/QuadraticCostFunction.h>
#include <NeuralNetworks/CostFunctions/CrossEntropyCostFunction.h>

#include <map>

namespace nnt
{
	static constexpr MathDomain md = MathDomain::Double;
	
	class NetworkTests : public ::testing::Test
	{
	public:
		const std::map<MathDomain, std::string> extension = { {MathDomain::Float, "Single"},
														      {MathDomain::Double, "Double"}};
		
		template<MathDomain T>
		nn::TrainingData<T> GetData(const std::string& fileType, const size_t nRowsInput, const size_t nRowsOutput, const size_t nCols)
		{
			const std::string path = getenv("DATA_PATH");
			
			auto input = cl::MatrixFromBinaryFile<MemorySpace::Device, T>(path + "/Data/" + fileType + "Input" + extension.at(md) + ".npy", true);
			if (input.nRows() != nRowsInput) std::abort();
			if (input.nCols() != nCols) std::abort();
			
			auto output = cl::MatrixFromBinaryFile<MemorySpace::Device, T>(path + "/Data/" + fileType + "Output" + extension.at(md) + ".npy", true);
			if (output.nRows() != nRowsOutput) std::abort();
			if (output.nCols() != nCols) std::abort();
			
			return nn::TrainingData<T>(std::move(input), std::move(output));
		}
	};
	
	TEST_F(NetworkTests, TrivialNetworkConsistency)
	{
		auto trainingData = GetData<md>("Training", 784, 10, 50000);
		auto validationData = GetData<md>("Validation", 784, 10, 10000);
		auto testData = GetData<md>("Test", 784, 10, 10000);
		
		nn::Vector<MathDomain::Int> cache1(10000u);
		nn::Vector<MathDomain::Int> cache2(10000u);
		nn::Vector<MathDomain::Int> cache3(10000u);
		size_t currentIter = 0;
		
		std::vector<int> expectedScores = { 9067, 9204, 9241 };
		std::function<double(nn::Matrix<md>&, const nn::Matrix<md>&)> evaluator = [&](nn::Matrix<md>& modelOutput, const nn::Matrix<md>& expectedOutput)
		{
			assert(modelOutput.nCols() == cache1.size());
			modelOutput.ColumnWiseArgAbsMaximum(cache1);
			
			assert(expectedOutput.nCols() == cache2.size());
			expectedOutput.ColumnWiseArgAbsMaximum(cache2);
			
			int score = cache1.CountEquals(cache2, cache3.GetBuffer());
			
			if (expectedScores[currentIter++] != score)
				throw std::runtime_error("");
			
			return static_cast<double>(score);
		};
		
		nn::NetworkTrainingData<md> data(trainingData, testData, validationData, evaluator);
		data.debugLevel = 2;
		data.epochCalculationAccuracyTestData = 1;
		
		data.hyperParameters.nEpochs = 3;
		data.hyperParameters.miniBacthSize = 10;
		data.hyperParameters.learningRate = 3.0;
		data.hyperParameters.lambda = 0.0;
		
		auto networkTopology = std::vector<size_t>{{ 784, 30, 10 }};
		nn::Network<md> network(networkTopology,
				                nn::RandomBiasWeightInitializer<md>(),
				                std::make_unique<nn::QuadraticCostFunction<md>>(),
				                std::make_unique<nn::RandomShuffler<md>>());
		
		ASSERT_NO_THROW(network.Train(data));
	}
	
	TEST_F(NetworkTests, TrivialNetworkSmallVarianceWeightsConsistency)
	{
		auto trainingData = GetData<md>("Training", 784, 10, 50000);
		auto validationData = GetData<md>("Validation", 784, 10, 10000);
		auto testData = GetData<md>("Test", 784, 10, 10000);
		
		nn::Vector<MathDomain::Int> cache1(10000u);
		nn::Vector<MathDomain::Int> cache2(10000u);
		nn::Vector<MathDomain::Int> cache3(10000u);
		size_t currentIter = 0;
		
		std::vector<int> expectedScores = { 9393, 9393, 9482 };
		std::function<double(nn::Matrix<md>&, const nn::Matrix<md>&)> evaluator = [&](nn::Matrix<md>& modelOutput, const nn::Matrix<md>& expectedOutput)
		{
			assert(modelOutput.nCols() == cache1.size());
			modelOutput.ColumnWiseArgAbsMaximum(cache1);
			
			assert(expectedOutput.nCols() == cache2.size());
			expectedOutput.ColumnWiseArgAbsMaximum(cache2);
			
			int score = cache1.CountEquals(cache2, cache3.GetBuffer());
			
			if (expectedScores[currentIter++] != score)
				throw std::runtime_error("");
			
			return static_cast<double>(score);
		};
		
		nn::NetworkTrainingData<md> data(trainingData, testData, validationData, evaluator);
		data.debugLevel = 2;
		data.epochCalculationAccuracyTestData = 1;
		data.nMaxEpochsWithNoScoreImprovements = 3;
		
		data.hyperParameters.nEpochs = 3;
		data.hyperParameters.miniBacthSize = 10;
		data.hyperParameters.learningRate = 3.0;
		data.hyperParameters.lambda = 0.0;
		
		auto networkTopology = std::vector<size_t>{{ 784, 30, 10 }};
		nn::Network<md> network(networkTopology,
				                nn::SmallVarianceRandomBiasWeightInitializer<md>(),
				                std::make_unique<nn::QuadraticCostFunction<md>>(),
				                std::make_unique<nn::RandomShuffler<md>>());
		ASSERT_NO_THROW(network.Train(data));
	}
	
	TEST_F(NetworkTests, TrivialNetworkSmallVarianceWeightsRegularizedConsistency)
	{
		auto trainingData = GetData<md>("Training", 784, 10, 50000);
		auto validationData = GetData<md>("Validation", 784, 10, 10000);
		auto testData = GetData<md>("Test", 784, 10, 10000);
		
		nn::Vector<MathDomain::Int> cache1(10000u);
		nn::Vector<MathDomain::Int> cache2(10000u);
		nn::Vector<MathDomain::Int> cache3(10000u);
		size_t currentIter = 0;
		
		std::vector<int> expectedScores = { 9213, 9376, 9451 };
		std::function<double(nn::Matrix<md>&, const nn::Matrix<md>&)> evaluator = [&](nn::Matrix<md>& modelOutput, const nn::Matrix<md>& expectedOutput)
		{
			assert(modelOutput.nCols() == cache1.size());
			modelOutput.ColumnWiseArgAbsMaximum(cache1);
			
			assert(expectedOutput.nCols() == cache2.size());
			expectedOutput.ColumnWiseArgAbsMaximum(cache2);
			
			int score = cache1.CountEquals(cache2, cache3.GetBuffer());
			if (expectedScores[currentIter++] != score)
				throw std::runtime_error(std::to_string(score));
			
			return static_cast<double>(score);
		};
		
		nn::NetworkTrainingData<md> data(trainingData, testData, validationData, evaluator);
		data.debugLevel = 2;
		data.epochCalculationAccuracyTestData = 1;
		data.nMaxEpochsWithNoScoreImprovements = 3;
		
		data.hyperParameters.nEpochs = 3;
		data.hyperParameters.miniBacthSize = 10;
		data.hyperParameters.learningRate = 0.5;
		data.hyperParameters.lambda = 0.1;
		
		auto networkTopology = std::vector<size_t>{{ 784, 30, 10 }};
		nn::Network<md> network(networkTopology,
								nn::SmallVarianceRandomBiasWeightInitializer<md>(),
								        std::make_unique<nn::QuadraticCostFunction<md>>(),
								        std::make_unique<nn::RandomShuffler<md>>());
		ASSERT_NO_THROW(network.Train(data));
	}
	
	TEST_F(NetworkTests, TrivialNetworkSmallVarianceWeightsRegularizedCrossEntropyConsistency)
	{
		auto trainingData = GetData<md>("Training", 784, 10, 50000);
		auto validationData = GetData<md>("Validation", 784, 10, 10000);
		auto testData = GetData<md>("Test", 784, 10, 10000);
		
		nn::Vector<MathDomain::Int> cache1(10000u);
		nn::Vector<MathDomain::Int> cache2(10000u);
		nn::Vector<MathDomain::Int> cache3(10000u);
		size_t currentIter = 0;
		
		std::vector<int> expectedScores = { 9416, 9448, 9516 };
		std::function<double(nn::Matrix<md>&, const nn::Matrix<md>&)> evaluator = [&](nn::Matrix<md>& modelOutput, const nn::Matrix<md>& expectedOutput)
		{
			assert(modelOutput.nCols() == cache1.size());
			modelOutput.ColumnWiseArgAbsMaximum(cache1);
			
			assert(expectedOutput.nCols() == cache2.size());
			expectedOutput.ColumnWiseArgAbsMaximum(cache2);
			
			int score = cache1.CountEquals(cache2, cache3.GetBuffer());
			
			if (expectedScores[currentIter++] != score)
				throw std::runtime_error(std::to_string(score));
			
			return static_cast<double>(score);
		};
		
		nn::NetworkTrainingData<md> data(trainingData, testData, validationData, evaluator);
		data.debugLevel = 2;
		data.epochCalculationAccuracyTestData = 1;
		data.nMaxEpochsWithNoScoreImprovements = 3;
		
		data.hyperParameters.nEpochs = 3;
		data.hyperParameters.miniBacthSize = 10;
		data.hyperParameters.learningRate = 0.5;
		data.hyperParameters.lambda = 0.1;
		
		auto networkTopology = std::vector<size_t>{{ 784, 30, 10 }};
		nn::Network<md> network(networkTopology,
				                nn::SmallVarianceRandomBiasWeightInitializer<md>(),
				                std::make_unique<nn::CrossEntropyCostFunction<md>>(),
				                std::make_unique<nn::RandomShuffler<md>>());
		ASSERT_NO_THROW(network.Train(data));
	}
}