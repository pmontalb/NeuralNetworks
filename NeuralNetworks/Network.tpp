
#include <NeuralNetworks/Initializers/SmallVarianceRandomBiasWeightInitializer.h>
#include <NeuralNetworks/CostFunctions/CrossEntropyCostFunction.h>
#include <NeuralNetworks/Shufflers/RandomShuffler.h>

namespace nn
{
	template<MathDomain mathDomain>
	Network<mathDomain>::Network(const std::vector<size_t>& nNeurons) noexcept
		: Network(nNeurons,
				  SmallVarianceRandomBiasWeightInitializer<mathDomain>(),
				  std::make_unique<CrossEntropyCostFunction<mathDomain>>(),
				  std::make_unique<RandomShuffler<mathDomain>>())
	{
	}
	
	template<MathDomain mathDomain>
	Network<mathDomain>::Network(const std::vector<lay>& layers,
								 std::unique_ptr<ICostFunction<mathDomain>>&& costFunction,
								 std::unique_ptr<IShuffler<mathDomain>>&& miniBatchShuffler) noexcept
			: _layers(layers), _costFunction(std::move(costFunction)), _miniBatchShuffler(std::move(miniBatchShuffler))
	{
		for (size_t l = 1; l < _layers.size(); ++l)
			assert(_layers[l]->GetNumberOfInputs() == _layers[l - 1]->GetNumberOfOutputs());
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
	void Network<mathDomain>::Train(const NetworkTrainingData<mathDomain>& networkTrainingData) noexcept
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
				
				const double totalCost =  _costFunction->Evaluate(modelOutput->second, networkData.expectedOutput, _layers, networkTrainingData.hyperParameters.lambda);
				std::cout << "\t###\tTotal Cost = " << totalCost << " ###" << std::endl;
			}
		};
		
		size_t nMiniBatchIterations = networkTrainingData.trainingData.GetNumberOfSamples() / networkTrainingData.hyperParameters.miniBacthSize;
		for (size_t i = 0; i < networkTrainingData.hyperParameters.nEpochs; ++i)
		{
			if (networkTrainingData.debugLevel > 0)
				std::cout << "Epoch " << i << " start..." << std::endl;
			
			sw.Start();
			
			_miniBatchShuffler->Shuffle(data.networkTrainingData.trainingData.input, data.networkTrainingData.trainingData.expectedOutput);
			
			data.endIndex = 0;
			for (size_t n = 0; n < nMiniBatchIterations; ++n)
			{
				data.startIndex = data.endIndex;
				data.endIndex += networkTrainingData.hyperParameters.miniBacthSize;
				
				// TODO: put in the optimizer
				UpdateMiniBatch(data);
			}
			
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
	
	template<MathDomain mathDomain>
	void Network<mathDomain>::UpdateMiniBatch(MiniBatchData<mathDomain>& data) noexcept
	{
		Stopwatch sw(true);
		for (auto& layer: _layers)
			layer->Reset();
		
		// calculates analytically the gradient, by means of backward differentiation
		AdjointDifferentiation(data);
		
		// update biases and weights with the previously calculated gradient
		UpdateBiasesAndWeights(data);
		
		sw.Stop();
		
		if (data.networkTrainingData.debugLevel > 2)
			std::cout << "\tMiniBatch[" << data.startIndex << ", " << data.endIndex << "] completed in " << sw.GetMilliSeconds() << "ms" << std::endl;
	}
	
	template<MathDomain mathDomain>
	void Network<mathDomain>::AdjointDifferentiation(MiniBatchData<mathDomain>& data) const noexcept
	{
		Stopwatch sw(true);
		
		for (size_t i = data.startIndex; i < data.endIndex; ++i)
		{
			// network evaluation
			_layers[0]->Evaluate(*data.networkTrainingData.trainingData.input.columns[i]);
			for (size_t l = 1; l < GetNumberOfLayers(); ++l)
				_layers[l]->Evaluate(_layers[l - 1]->GetActivation());
			
			// *** Back propagation of the last layer ***
			// last layer activation will be overridden with the cost function derivative!
			SetCostFunctionGradient(*data.networkTrainingData.trainingData.expectedOutput.columns[i]);
			const auto& costFunctionGradient = _layers.back()->GetActivation();
			
			// back propagation
			_layers.back()->GetBiasGradient().AddEqual(costFunctionGradient);
			mat::KroneckerProduct(_layers.back()->GetWeightGradient(),
					              costFunctionGradient,
					              _layers[GetNumberOfLayers() - 2]->GetActivation());
			// ***
			
			// now back-propagate through the remaining layers
			for (size_t l = 2; l <= GetNumberOfLayers(); ++l)
			{
				auto& nextLayer     = _layers[GetNumberOfLayers() - l + 1];
				auto& layer         = _layers[GetNumberOfLayers() - l];
				auto& previousInput = l == 2 ? *data.networkTrainingData.trainingData.input.columns[i] : _layers[GetNumberOfLayers() - l - 1]->GetActivation();
				
				nextLayer->GetWeight().Dot(layer->GetBiasGradientCache(),
						                   l == 2 ? costFunctionGradient : nextLayer->GetBiasGradientCache(),
						                   MatrixOperation::Transpose);
				
				layer->GetBiasGradientCache() %= layer->GetActivationGradient();
				layer->GetBiasGradient() += layer->GetBiasGradientCache();
				mat::KroneckerProduct(layer->GetWeightGradient(), layer->GetBiasGradientCache(), previousInput);
			}
		}
		
		sw.Stop();
		
		if (data.networkTrainingData.debugLevel > 3)
			std::cout << "\t\tAD[" << data.startIndex << ", " << data.endIndex << "] completed in " << sw.GetMilliSeconds() << "ms" << std::endl;
	}
	
	template<MathDomain mathDomain>
	void Network<mathDomain>::SetCostFunctionGradient(const vec& expectedOuput) const noexcept
	{
		auto& lastLayer = _layers.back();
		auto& modelOutput = lastLayer->GetActivation();  // this will be overridden!
		assert(modelOutput.size() == expectedOuput.size());
		
		_costFunction->EvaluateGradient(modelOutput, expectedOuput, lastLayer->GetActivationGradient());
	}
	
	template<MathDomain mathDomain>
	void Network<mathDomain>::UpdateBiasesAndWeights(MiniBatchData<mathDomain>& data) noexcept
	{
		Stopwatch sw(true);
		
		const double averageLearningRate = data.networkTrainingData.hyperParameters.GetAverageLearningRate();
		const double regularizationFactor = 1.0 - (data.networkTrainingData.hyperParameters.learningRate * data.networkTrainingData.hyperParameters.lambda) / data.networkTrainingData.trainingData.GetNumberOfSamples();
		for (auto& layer: _layers)
			layer->Update(averageLearningRate, regularizationFactor);
		
		sw.Stop();
		
		if (data.networkTrainingData.debugLevel > 3)
			std::cout << "\t\tUBW[" << data.startIndex << ", " << data.endIndex << "] completed in " << sw.GetMilliSeconds() << "ms" << std::endl;
	}
}
