
namespace nn
{
	template<MathDomain mathDomain>
	Network<mathDomain>::Network(const std::vector<size_t>& nNeurons) noexcept
		: _nNeurons(nNeurons)
	{
		for (size_t l = 0; l < GetNumberOfLayers(); ++l)
		{
			_biases.emplace_back(vec(_nNeurons[l + 1], 0.0));
			_biases.back().RandomGaussian();
			
			_weights.emplace_back(mat(_nNeurons[l + 1], _nNeurons[l], 0.0));
			_weights.back().RandomGaussian();
		}
	}
	
	template<MathDomain mathDomain>
	void Network<mathDomain>::Evaluate(mat& out, const NetworkTrainingData<mathDomain>& networkTrainingData, std::vector<vec>& cache) const noexcept
	{
		Stopwatch sw(true);
		
		if (cache.empty())
		{
			for (size_t l = 0; l < GetNumberOfLayers(); ++l)
				cache.emplace_back(vec(_nNeurons[l + 1], 0.0));
		}
		
		for (size_t col = 0; col < out.nCols(); ++col)
		{
			// sigmoid(w * cache + b)
			_weights[0].Dot(cache[0], *networkTrainingData.testData.input.columns[col]);
			cache[0].AddEqual(_biases[0]);
			nn::detail::Sigmoid(cache[0].GetBuffer(), cache[0].GetBuffer());
			
			// sigmoid(w * cache + b)
			for (size_t layer = 1; layer < GetNumberOfLayers(); ++layer)
			{
				_weights[layer].Dot(cache[layer], cache[layer - 1]);
				cache[layer].AddEqual(_biases[layer]);
				nn::detail::Sigmoid(cache[layer].GetBuffer(), cache[layer].GetBuffer());
			}
			
			// copy result in output
			out.columns[col]->ReadFrom(cache.back());
		}
		
		sw.Stop();
		if (networkTrainingData.debugLevel > 1)
			std::cout << "\tEvaluation completed in " << sw.GetMilliSeconds() << " ms" << std::endl;
	}
	
	template<MathDomain mathDomain>
	void Network<mathDomain>::Train(const NetworkTrainingData<mathDomain>& networkTrainingData) noexcept
	{
		Stopwatch sw;
		
		MiniBatchData<mathDomain> data(networkTrainingData, _biases, _weights);
		mat modelOutput(networkTrainingData.testData.expectedOutput.nRows(), networkTrainingData.testData.expectedOutput.nCols());
		
		std::vector<vec> evaluatorCache {};
		size_t nMiniBatchIterations = networkTrainingData.trainingData.GetNumberOfSamples() / networkTrainingData.hyperParameters.miniBacthSize;
		for (size_t i = 0; i < networkTrainingData.hyperParameters.nEpochs; ++i)
		{
			if (networkTrainingData.debugLevel > 0)
				std::cout << "Epoch " << i << " start..." << std::endl;
			
			sw.Start();
			
			cl::RandomShuffleColumnsPair(data.networkTrainingData.trainingData.input, data.networkTrainingData.trainingData.expectedOutput);
			
			data.endIndex = 0;
			for (size_t n = 0; n < nMiniBatchIterations; ++n)
			{
				data.startIndex = data.endIndex;
				data.endIndex += networkTrainingData.hyperParameters.miniBacthSize;
				
				UpdateMiniBatch(data);
			}
			
			if ((i + 1) % networkTrainingData.epochCalculation == 0)
			{
				Evaluate(modelOutput, networkTrainingData, evaluatorCache);
				networkTrainingData.evaluator(modelOutput, networkTrainingData.testData.expectedOutput);
			}
			
			sw.Stop();
			if (networkTrainingData.debugLevel > 0)
				std::cout << "Epoch " << i << " completed in " << sw.GetMilliSeconds() << "ms" << std::endl;
		}
	}
	
	template<MathDomain mathDomain>
	void Network<mathDomain>::UpdateMiniBatch(MiniBatchData<mathDomain>& data) noexcept
	{
		Stopwatch sw(true);
		data.Reset();
		
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
			for (size_t layer = 0; layer < GetNumberOfLayers(); ++layer)
			{
				data.EvaluateWorker(layer, layer == 0 ? *data.networkTrainingData.trainingData.input.columns[i]
				                                      : data.activations[layer - 1], _biases[layer], _weights[layer]);
			}
			
			// data.activations[GetNumberOfLayers() - 1] will be overridden with the objective function derivative
			SetObjectiveFunctionDerivative(i, data);
			
			// back propagation
			data.biasGradient[GetNumberOfLayers() - 1].AddEqual(data.activations[GetNumberOfLayers() - 1]);
			mat::KroneckerProduct(
					data.weightGradient[GetNumberOfLayers() - 1],
					data.activations[GetNumberOfLayers() - 1],
					data.activations[GetNumberOfLayers() - 2]);
			
			for (size_t layer = 2; layer <= GetNumberOfLayers(); ++layer)
			{
				_weights[GetNumberOfLayers() - layer + 1].Dot(
						data.biasGradientCache[GetNumberOfLayers() - layer],
						layer == 2 ? data.activations[GetNumberOfLayers() - 1] : data.biasGradientCache[GetNumberOfLayers() - layer + 1],
						MatrixOperation::Transpose);
				
				data.biasGradientCache[GetNumberOfLayers() - layer] %= data.activationsDerivative[GetNumberOfLayers() - layer];
				data.biasGradient[GetNumberOfLayers() - layer] += data.biasGradientCache[GetNumberOfLayers() - layer];
				
				data.weightGradientCache[GetNumberOfLayers() - layer].Set(0.0);
				mat::KroneckerProduct(
						data.weightGradient[GetNumberOfLayers() - layer],
						data.biasGradientCache[GetNumberOfLayers() - layer],
						layer == 2 ? *data.networkTrainingData.trainingData.input.columns[i] : data.activations[GetNumberOfLayers() - layer - 1]);
			}
		}
		
		sw.Stop();
		
		if (data.networkTrainingData.debugLevel > 3)
			std::cout << "\t\tAD[" << data.startIndex << ", " << data.endIndex << "] completed in " << sw.GetMilliSeconds() << "ms" << std::endl;
	}
	
	template<MathDomain mathDomain>
	void Network<mathDomain>::SetObjectiveFunctionDerivative(const size_t i, MiniBatchData<mathDomain>& data) const noexcept
	{
		vec& expected = data.activations[GetNumberOfLayers() - 1];
		const vec& actual = *data.networkTrainingData.trainingData.expectedOutput.columns[i];
		
		assert(expected.size() == actual.size());
		
		expected -= actual;
		expected %= data.activationsDerivative[GetNumberOfLayers() - 1];
	}
	
	template<MathDomain mathDomain>
	void Network<mathDomain>::UpdateBiasesAndWeights(MiniBatchData<mathDomain>& data) noexcept
	{
		Stopwatch sw(true);
		
		const double averageLearningRate = data.networkTrainingData.hyperParameters.GetAverageLearningRate();
		for (size_t i = 0; i < _biases.size(); ++i)
			_biases[i].AddEqual(data.biasGradient[i], -averageLearningRate);

		for (size_t i = 0; i < _weights.size(); ++i)
			_weights[i].AddEqual(data.weightGradient[i], -averageLearningRate);
		
		sw.Stop();
		
		if (data.networkTrainingData.debugLevel > 3)
			std::cout << "\t\tUBW[" << data.startIndex << ", " << data.endIndex << "] completed in " << sw.GetMilliSeconds() << "ms" << std::endl;
	}
}