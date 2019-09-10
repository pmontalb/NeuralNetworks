
#include <NeuralNetworks/Layers/Initializers/SmallVarianceRandomBiasWeightInitializer.h>
#include <NeuralNetworks/CostFunctions/CrossEntropyCostFunctionSigmoid.h>
#include <NeuralNetworks/Optimizers/IOptimizer.h>

namespace nn
{
	template<MathDomain mathDomain>
	Network<mathDomain>::Network(std::vector<lay>&& layers) noexcept
	{
		for (auto&& layer: layers)
			_layers.emplace_back(std::move(layer));
		
		for (size_t l = 1; l < _layers.size(); ++l)
			assert(_layers[l]->GetNumberOfInputs() == _layers[l - 1]->GetNumberOfOutputs());
	}
	
	template<MathDomain mathDomain>
	Network<mathDomain>::Network(std::istream& stream) noexcept
	{
		*this >> stream;
	}
	
	template<MathDomain mathDomain>
	std::ostream& Network<mathDomain>::operator <<(std::ostream& stream) const noexcept
	{
		stream << _layers.size() << std::endl;
		for (const auto& layer: _layers)
			*layer << stream;
	
		return stream;
	}
	
	template<MathDomain mathDomain>
	std::istream& Network<mathDomain>::operator >>(std::istream& stream) noexcept
	{
		_layers.clear();
		
		std::string line;
		
		std::getline(stream, line);
		const size_t nLayers = static_cast<size_t>(std::atoi(line.c_str()));
		
		for (size_t l = 0; l < nLayers; ++l)
		{
			std::getline(stream, line);
			const LayerType type = GetLayerType(line);
			
			std::getline(stream, line);
			const size_t nInput = static_cast<size_t>(std::atoi(line.c_str()));
			
			std::getline(stream, line);
			const size_t nOutput = static_cast<size_t>(std::atoi(line.c_str()));
			
			std::getline(stream, line);
			const ActivationFunctionType activationFunctionType = GetActivationFunctionType(line);
			auto activationFunction = ActivationFunctionFactory<mathDomain>::Create(activationFunctionType);
			
			auto layer = LayerFactory<mathDomain>::Create(type, nInput, nOutput, std::move(activationFunction),
			                                              std::move(TrivialBiasWeightInitializer<mathDomain>()));
			*layer >> stream;
			
			_layers.emplace_back(std::move(layer));
		}
	
		return stream;
	}
	
	template<MathDomain mathDomain>
	void Network<mathDomain>::Evaluate(mat& out, const mat& in, const int debugLevel) const noexcept
	{
		Stopwatch sw(true);
		
		for (size_t col = 0; col < out.nCols(); ++col)
		{
			// use input for first layer
			_layers.front()->Evaluate(*in.columns[col]);
			
			for (size_t l = 1; l < GetNumberOfLayers() - 1; ++l)
				_layers[l]->Evaluate(_layers[l - 1]->GetActivation());
			
			// use output column for last layer!
			_layers.back()->Evaluate(_layers[GetNumberOfLayers() - 2]->GetActivation(), out.columns[col].get());
		}
		
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
				const auto modelOutput = modelOutputCache.insert({ networkData.expectedOutput.nCols(), mat(networkData.expectedOutput.nRows(), networkData.expectedOutput.nCols()) }).first;
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
				
				const double totalCost =  optimizer.GetCostFunction().Evaluate(modelOutput->second, networkData.expectedOutput, _layers, networkTrainingData.hyperParameters.lambda);
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
