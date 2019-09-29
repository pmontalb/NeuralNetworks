
#include <NeuralNetworks/Layers/Initializers/SmallVarianceRandomBiasWeightInitializer.h>
#include <NeuralNetworks/CostFunctions/CrossEntropyCostFunction.h>
#include <NeuralNetworks/Optimizers/IOptimizer.h>

namespace nn
{
	template<MathDomain mathDomain>
	Network<mathDomain>::Network(NetworkTopology<mathDomain>&& topology) noexcept
		: _topology(std::move(topology))
	{
	}
	
	template<MathDomain mathDomain>
	Network<mathDomain>::Network(std::istream& stream) noexcept
		: _topology(stream)
	{
	}
	
	template<MathDomain mathDomain>
	std::ostream& Network<mathDomain>::operator <<(std::ostream& stream) const noexcept
	{
		_topology << stream;
		return stream;
	}
	
	template<MathDomain mathDomain>
	std::istream& Network<mathDomain>::operator >>(std::istream& stream) noexcept
	{
		_topology >> stream;
		return stream;
	}
	
	template<MathDomain mathDomain>
	void Network<mathDomain>::Evaluate(mat& out, const mat& in, const int debugLevel) const noexcept
	{
		Stopwatch sw(true);
		
		_topology.Evaluate(in, false, &out);
		
		sw.Stop();
		if (debugLevel > 1)
			std::cout << "\tEvaluation completed in " << sw.GetMilliSeconds() << " ms" << std::endl;
	}
	
	template<MathDomain mathDomain>
	void Network<mathDomain>::Train(IOptimizer<mathDomain>& optimizer, const NetworkTrainingData<mathDomain>& networkTrainingData) noexcept
	{
		Stopwatch sw;
		
		MiniBatchData<mathDomain> data(networkTrainingData);
		std::unordered_map<size_t, mat> modelOutputCache;
		static constexpr size_t nEvaluationDimensions = { 3 };
		std::array<double, nEvaluationDimensions> bestAccuracies = {{ 0.0 }};
		std::array<size_t, nEvaluationDimensions> nEpochsWithNoImprovements = {{ 0 }};
		const auto accuracyEvaluator = [&](const auto i, const auto epoch, const auto& networkData, auto accuracyIndex)
		{
			if (epoch > 0 && (i + 1) % epoch == 0)
			{
				auto modelOutput = modelOutputCache.find(networkData.expectedOutput.nCols());
				if (modelOutput == modelOutputCache.end())
					modelOutput = modelOutputCache.emplace(std::piecewise_construct,
							                               std::forward_as_tuple(networkData.expectedOutput.nCols()),
							                               std::forward_as_tuple(mat(networkData.expectedOutput.nRows(), networkData.expectedOutput.nCols()))).first;
				Evaluate(modelOutput->second, networkData.input, networkTrainingData.debugLevel);
				const double accuracy = networkTrainingData.evaluator(modelOutput->second, networkData.expectedOutput);
				
				if (accuracy > bestAccuracies[accuracyIndex])
				{
					std::cout << "\t***\t\t*NEW best accuracy (" << bestAccuracies[accuracyIndex] << " -> " << accuracy << ") ***" << std::endl;
					bestAccuracies[accuracyIndex] = accuracy;
					nEpochsWithNoImprovements[accuracyIndex] = 0;
				}
				else
				{
					++nEpochsWithNoImprovements[accuracyIndex];
					if (nEpochsWithNoImprovements[accuracyIndex] > networkTrainingData.nMaxEpochsWithNoScoreImprovements)
					{
						std::cout << "\t***\tEarly stop due to " << nEpochsWithNoImprovements[accuracyIndex] << " epochs with no improvements" << std::endl;
						return false;
					}
				}
			}
			
			return true;
		};
		
		const auto totalCostEvaluator = [&](const auto i, const auto epoch, const auto& networkData)
		{
			if (epoch > 0 && (i + 1) % epoch == 0)
			{
				const auto modelOutput = modelOutputCache.insert({ networkData.expectedOutput.nCols(), mat(networkData.expectedOutput.nRows(), networkData.expectedOutput.nCols()) }).first;
				Evaluate(modelOutput->second, networkData.input, networkTrainingData.debugLevel);
				
				const double totalCost =  optimizer.GetCostFunction().Evaluate(modelOutput->second, networkData.expectedOutput, _topology, networkTrainingData.hyperParameters.lambda);
				std::cout << "\t###\tTotal Cost = " << totalCost << " ###" << std::endl;
			}
		};
		
		for (size_t i = 0; i < networkTrainingData.hyperParameters.nEpochs; ++i)
		{
			if (networkTrainingData.debugLevel > 0)
				std::cout << "Epoch " << i << " start..." << std::endl;
			
			sw.Start();
			
			optimizer.Train(networkTrainingData);
			
			// accuracy and total cost -> mainly debug stuff!
			if (!accuracyEvaluator(i, networkTrainingData.epochCalculationAccuracyTestData, networkTrainingData.testData, 0u))
				return;
			if (!accuracyEvaluator(i, networkTrainingData.epochCalculationAccuracyValidationData, networkTrainingData.validationData, 1u))
				return;
			if (!accuracyEvaluator(i, networkTrainingData.epochCalculationAccuracyTrainingData, networkTrainingData.trainingData, 2u))
				return;
			totalCostEvaluator(i, networkTrainingData.epochCalculationTotalCostTestData, networkTrainingData.testData);
			totalCostEvaluator(i, networkTrainingData.epochCalculationTotalCostValidationData, networkTrainingData.validationData);
			totalCostEvaluator(i, networkTrainingData.epochCalculationTotalCostTrainingData, networkTrainingData.trainingData);
			//
			
			sw.Stop();
			if (networkTrainingData.debugLevel > 0)
				std::cout << "Epoch " << i << " completed in " << sw.GetMilliSeconds() << "ms" << std::endl;
		}
	}
}
