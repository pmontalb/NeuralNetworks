
#include <Network.h>
#include <NeuralNetworks/Layers/Initializers/All.h>
#include <NeuralNetworks/CostFunctions/All.h>
#include <NeuralNetworks/Layers/All.h>
#include <NeuralNetworks/Activations/All.h>
#include <NeuralNetworks/Optimizers/All.h>
#include <NeuralNetworks/Optimizers/Shufflers/All.h>

#include <map>
#include <fstream>
#include <gtest/gtest.h>

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
			
			auto input = cl::MatrixFromBinaryFile<MemorySpace::Device, T>(path + "/Data/" + fileType + "Input" + extension.at(md) + ".npy", false, true);
			if (input.nRows() != nRowsInput) std::abort();
			if (input.nCols() != nCols) std::abort();
			
			auto output = cl::MatrixFromBinaryFile<MemorySpace::Device, T>(path + "/Data/" + fileType + "Output" + extension.at(md) + ".npy", false,true);
			if (output.nRows() != nRowsOutput) std::abort();
			if (output.nCols() != nCols) std::abort();
			
			return nn::TrainingData<T>(std::move(input), std::move(output));
		}
	};
	
	TEST_F(NetworkTests, Serialization)
	{
		std::vector<std::unique_ptr<nn::ILayer<md>>> networkTopology;
		networkTopology.emplace_back(std::make_unique<nn::DenseLayer<md>>(784, 30, std::make_unique<nn::SigmoidActivationFunction<md>>(), nn::RandomBiasWeightInitializer<md>()));
		networkTopology.emplace_back(std::make_unique<nn::DenseLayer<md>>(30, 10, std::make_unique<nn::SigmoidActivationFunction<md>>(), nn::RandomBiasWeightInitializer<md>()));
		nn::Network<md> network(std::move(networkTopology));

		std::ofstream f("out");
		ASSERT_TRUE(f.is_open());
		network.Serialize(f);
		f.close();
		
		const auto& layers = network.GetLayers();
		
		std::ifstream g("out");
		ASSERT_TRUE(g.is_open());
		nn::Network<md> deserializedNetwork(g);
		const auto& deserializedLayers = deserializedNetwork.GetLayers();
		
		ASSERT_EQ(layers.size(), deserializedLayers.size());
		for (size_t i = 0; i < layers.size(); ++i)
		{
			ASSERT_EQ(layers[i]->GetType(), deserializedLayers[i]->GetType());
			ASSERT_EQ(layers[i]->GetNumberOfInputs(), deserializedLayers[i]->GetNumberOfInputs());
			
			ASSERT_EQ(layers[i]->GetWeight().nRows(), deserializedLayers[i]->GetWeight().nRows());
			ASSERT_EQ(layers[i]->GetWeight().nCols(), deserializedLayers[i]->GetWeight().nCols());
			auto weight = layers[i]->GetWeight().Get();
			auto deserializedWeight = deserializedLayers[i]->GetWeight().Get();
			for (size_t j = 0; j < weight.size(); ++j)
				ASSERT_DOUBLE_EQ(weight[j], deserializedWeight[j]);
			
			ASSERT_EQ(layers[i]->GetNumberOfOutputs(), deserializedLayers[i]->GetNumberOfOutputs());
			ASSERT_EQ(layers[i]->GetBias().size(), deserializedLayers[i]->GetBias().size());
			auto bias = layers[i]->GetBias().Get();
			auto deserializedBias = deserializedLayers[i]->GetBias().Get();
			for (size_t j = 0; j < bias.size(); ++j)
				ASSERT_DOUBLE_EQ(bias[j], deserializedBias[j]);
		}
		
		// verify that score is the same
		cl::ColumnWiseMatrix<MemorySpace::Device, md> out(layers.back()->GetNumberOfOutputs(), 1);
		cl::ColumnWiseMatrix<MemorySpace::Device, md> in(layers.front()->GetNumberOfInputs(), 1, 1.234);
		network.Evaluate(out, in, 99);
		
		cl::ColumnWiseMatrix<MemorySpace::Device, md> out2(layers.back()->GetNumberOfOutputs(), 1);
		deserializedNetwork.Evaluate(out2, in, 99);
		
		auto _out = out.Get();
		auto _out2 = out2.Get();
		ASSERT_EQ(out.size(), out2.size());
		for (size_t i = 0; i < out.size(); ++i)
			ASSERT_DOUBLE_EQ(_out[i], _out2[i]);
	}
	
	TEST_F(NetworkTests, TrivialNetworkConsistency)
	{
		auto trainingData = GetData<md>("Training", 784, 10, 50000);
		auto validationData = GetData<md>("Validation", 784, 10, 10000);
		auto testData = GetData<md>("Test", 784, 10, 10000);
		
		nn::Vector<MathDomain::Int> cache1(10000u);
		nn::Vector<MathDomain::Int> cache2(10000u);
		nn::Vector<MathDomain::Int> cache3(10000u);
		size_t currentIter = 0;
		
		std::vector<int> expectedScores = { 9052, 9190, 9251 };
		std::vector<int> actualScores;
		std::function<double(nn::Matrix<md>&, const nn::Matrix<md>&)> evaluator = [&](nn::Matrix<md>& modelOutput, const nn::Matrix<md>& expectedOutput)
		{
			assert(modelOutput.nCols() == cache1.size());
			modelOutput.ColumnWiseArgAbsMaximum(cache1);
			
			assert(expectedOutput.nCols() == cache2.size());
			expectedOutput.ColumnWiseArgAbsMaximum(cache2);
			
			int score = cache1.CountEquals(cache2, cache3.GetBuffer());
			
			actualScores.push_back(score);
			
			return static_cast<double>(score);
		};
		
		nn::NetworkTrainingData<md> data(trainingData, testData, validationData, evaluator);
		data.debugLevel = 2;
		data.epochCalculationAccuracyTestData = 1;
		
		data.hyperParameters.nEpochs = 3;
		data.hyperParameters.miniBacthSize = 10;
		data.hyperParameters.learningRate = 3.0;
		data.hyperParameters.lambda = 0.0;
		
		std::vector<std::unique_ptr<nn::ILayer<md>>> networkTopology;
		networkTopology.emplace_back(std::make_unique<nn::DenseLayer<md>>(784, 30, std::make_unique<nn::SigmoidActivationFunction<md>>(), nn::RandomBiasWeightInitializer<md>()));
		networkTopology.emplace_back(std::make_unique<nn::DenseLayer<md>>(30, 10, std::make_unique<nn::SigmoidActivationFunction<md>>(), nn::RandomBiasWeightInitializer<md>()));
		nn::Network<md> network(std::move(networkTopology));
		
		nn::BatchedSgd<md> optimizer(network.GetLayers(), std::make_unique<nn::QuadraticCostFunction<md>>(), std::make_unique<nn::RandomShuffler<md>>());
		network.Train(optimizer, data);
		
		for (size_t i = 0; i < actualScores.size(); ++i)
			ASSERT_DOUBLE_EQ(expectedScores[i], actualScores[i]);
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
		
		std::vector<int> expectedScores = { 9393, 9397, 9502 };
		std::vector<int> actualScores;
		std::function<double(nn::Matrix<md>&, const nn::Matrix<md>&)> evaluator = [&](nn::Matrix<md>& modelOutput, const nn::Matrix<md>& expectedOutput)
		{
			assert(modelOutput.nCols() == cache1.size());
			modelOutput.ColumnWiseArgAbsMaximum(cache1);
			
			assert(expectedOutput.nCols() == cache2.size());
			expectedOutput.ColumnWiseArgAbsMaximum(cache2);
			
			int score = cache1.CountEquals(cache2, cache3.GetBuffer());
			
			actualScores.push_back(score);
			
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
		
		std::vector<std::unique_ptr<nn::ILayer<md>>> networkTopology;
		networkTopology.emplace_back(std::make_unique<nn::DenseLayer<md>>(784, 30, std::make_unique<nn::SigmoidActivationFunction<md>>(), nn::SmallVarianceRandomBiasWeightInitializer<md>()));
		networkTopology.emplace_back(std::make_unique<nn::DenseLayer<md>>(30, 10, std::make_unique<nn::SigmoidActivationFunction<md>>(), nn::SmallVarianceRandomBiasWeightInitializer<md>()));
		nn::Network<md> network(std::move(networkTopology));
		
		nn::BatchedSgd<md> optimizer(network.GetLayers(), std::make_unique<nn::QuadraticCostFunction<md>>(), std::make_unique<nn::RandomShuffler<md>>());
		network.Train(optimizer, data);
		for (size_t i = 0; i < actualScores.size(); ++i)
			EXPECT_DOUBLE_EQ(expectedScores[i], actualScores[i]);
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
		std::vector<int> actualScores;
		std::function<double(nn::Matrix<md>&, const nn::Matrix<md>&)> evaluator = [&](nn::Matrix<md>& modelOutput, const nn::Matrix<md>& expectedOutput)
		{
			assert(modelOutput.nCols() == cache1.size());
			modelOutput.ColumnWiseArgAbsMaximum(cache1);
			
			assert(expectedOutput.nCols() == cache2.size());
			expectedOutput.ColumnWiseArgAbsMaximum(cache2);
			
			int score = cache1.CountEquals(cache2, cache3.GetBuffer());
			actualScores.push_back(score);
			
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
		
		std::vector<std::unique_ptr<nn::ILayer<md>>> networkTopology;
		networkTopology.emplace_back(std::make_unique<nn::DenseLayer<md>>(784, 30, std::make_unique<nn::SigmoidActivationFunction<md>>(), nn::SmallVarianceRandomBiasWeightInitializer<md>()));
		networkTopology.emplace_back(std::make_unique<nn::DenseLayer<md>>(30, 10, std::make_unique<nn::SigmoidActivationFunction<md>>(), nn::SmallVarianceRandomBiasWeightInitializer<md>()));
		nn::Network<md> network(std::move(networkTopology));
		
		nn::BatchedSgd<md> optimizer(network.GetLayers(), std::make_unique<nn::QuadraticCostFunction<md>>(), std::make_unique<nn::RandomShuffler<md>>());
		network.Train(optimizer, data);
		for (size_t i = 0; i < actualScores.size(); ++i)
			EXPECT_DOUBLE_EQ(expectedScores[i], actualScores[i]);
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
		std::vector<int> actualScores;
		std::function<double(nn::Matrix<md>&, const nn::Matrix<md>&)> evaluator = [&](nn::Matrix<md>& modelOutput, const nn::Matrix<md>& expectedOutput)
		{
			assert(modelOutput.nCols() == cache1.size());
			modelOutput.ColumnWiseArgAbsMaximum(cache1);
			
			assert(expectedOutput.nCols() == cache2.size());
			expectedOutput.ColumnWiseArgAbsMaximum(cache2);
			
			int score = cache1.CountEquals(cache2, cache3.GetBuffer());
			
			actualScores.push_back(score);
			
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
		
		std::vector<std::unique_ptr<nn::ILayer<md>>> networkTopology;
		networkTopology.emplace_back(std::make_unique<nn::DenseLayer<md>>(784, 30, std::make_unique<nn::SigmoidActivationFunction<md>>(), nn::SmallVarianceRandomBiasWeightInitializer<md>()));
		networkTopology.emplace_back(std::make_unique<nn::DenseLayer<md>>(30, 10, std::make_unique<nn::SigmoidActivationFunction<md>>(), nn::SmallVarianceRandomBiasWeightInitializer<md>()));
		nn::Network<md> network(std::move(networkTopology));
		
		nn::BatchedSgd<md> optimizer(network.GetLayers(), std::make_unique<nn::CrossEntropyCostFunctionSigmoid<md>>(), std::make_unique<nn::RandomShuffler<md>>());
		network.Train(optimizer, data);
		for (size_t i = 0; i < actualScores.size(); ++i)
			EXPECT_DOUBLE_EQ(expectedScores[i], actualScores[i]);
	}
	
	TEST_F(NetworkTests, RefinedNetworkConsistency)
	{
		auto trainingData = GetData<md>("Training", 784, 10, 50000);
		auto validationData = GetData<md>("Validation", 784, 10, 10000);
		auto testData = GetData<md>("Test", 784, 10, 10000);
		
		nn::Vector<MathDomain::Int> cache1(10000u);
		nn::Vector<MathDomain::Int> cache2(10000u);
		nn::Vector<MathDomain::Int> cache3(10000u);
		size_t currentIter = 0;
		
		std::vector<int> expectedScores = { 9167, 9330, 9447 };
		std::vector<int> actualScores;
		std::function<double(nn::Matrix<md>&, const nn::Matrix<md>&)> evaluator = [&](nn::Matrix<md>& modelOutput, const nn::Matrix<md>& expectedOutput)
		{
			assert(modelOutput.nCols() == cache1.size());
			modelOutput.ColumnWiseArgAbsMaximum(cache1);
			
			assert(expectedOutput.nCols() == cache2.size());
			expectedOutput.ColumnWiseArgAbsMaximum(cache2);
			
			int score = cache1.CountEquals(cache2, cache3.GetBuffer());
			
			actualScores.push_back(score);
			
			return static_cast<double>(score);
		};
		
		nn::NetworkTrainingData<md> data(trainingData, testData, validationData, evaluator);
		data.debugLevel = 2;
		data.epochCalculationAccuracyTestData = 1;
		data.nMaxEpochsWithNoScoreImprovements = 3;
		
		data.hyperParameters.nEpochs = 3;
		data.hyperParameters.miniBacthSize = 10;
		data.hyperParameters.learningRate = 0.1;
		data.hyperParameters.lambda = 5.0;
		
		std::vector<std::unique_ptr<nn::ILayer<md>>> networkTopology;
		networkTopology.emplace_back(std::make_unique<nn::DenseLayer<md>>(784, 30, std::make_unique<nn::SigmoidActivationFunction<md>>(), nn::SmallVarianceRandomBiasWeightInitializer<md>()));
		networkTopology.emplace_back(std::make_unique<nn::SoftMaxLayer<md>>(30, 10, std::make_unique<nn::SoftMaxActivationFunction<md>>(), nn::ZeroBiasWeightInitializer<md>()));
		nn::Network<md> network(std::move(networkTopology));
		
		nn::BatchedSgd<md> optimizer(network.GetLayers(), network.GetLayers().back()->GetCrossEntropyCostFunction(), std::make_unique<nn::RandomShuffler<md>>());
		network.Train(optimizer, data);
		for (size_t i = 0; i < actualScores.size(); ++i)
			EXPECT_DOUBLE_EQ(expectedScores[i], actualScores[i]);
	}
}