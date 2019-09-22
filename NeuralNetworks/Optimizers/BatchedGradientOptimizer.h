#pragma once

#include <NeuralNetworks/Optimizers/GradientOptimizer.h>
#include <NeuralNetworks/Optimizers/Shufflers/IShuffler.h>

namespace nn
{
	template<MathDomain mathDomain>
	class BatchedGradientOptimizer: public GradientOptimizer<mathDomain>
	{
	public:
		BatchedGradientOptimizer(const NetworkTopology<mathDomain>& topology,
		                         const size_t miniBatchSize,
		                         std::unique_ptr<ICostFunction<mathDomain>>&& costFunction,
		                         std::unique_ptr<IShuffler<mathDomain>>&& miniBatchShuffler) noexcept
				: GradientOptimizer<mathDomain>(topology, std::move(costFunction)), _miniBatchSize(miniBatchSize), _miniBatchShuffler(std::move(miniBatchShuffler))
		{
		}
		
		void Train(const NetworkTrainingData<mathDomain>& networkTrainingData) noexcept override
		{
			_miniBatchShuffler->Shuffle(networkTrainingData.trainingData.input,
					                    networkTrainingData.trainingData.expectedOutput);
			
			MiniBatchData<mathDomain> batchData(networkTrainingData);
			Stopwatch sw;
			
			const size_t nMiniBatchIterations = networkTrainingData.trainingData.GetNumberOfSamples() / networkTrainingData.hyperParameters.miniBatchSize;
			for (size_t n = 0; n < nMiniBatchIterations; ++n)
			{
				batchData.startIndex = batchData.endIndex;
				batchData.endIndex += networkTrainingData.hyperParameters.miniBatchSize;
				batchData.endIndex = std::min(networkTrainingData.trainingData.GetNumberOfSamples(), batchData.endIndex);
				
				sw.Start();
				
				TrainMiniBatch(batchData);
				UpdateLayers(batchData);
				
				sw.Stop();
				if (batchData.networkTrainingData.debugLevel > 2)
					std::cout << "\tMiniBatch[" << batchData.startIndex << ", " << batchData.endIndex << "] completed in " << sw.GetMilliSeconds() << "ms" << std::endl;
			}
		}
		
	protected:
		virtual void TrainMiniBatch(MiniBatchData<mathDomain>& batchData) noexcept = 0;
		
	private:
		void UpdateLayers(MiniBatchData<mathDomain>& batchData) const noexcept
		{
			Stopwatch sw(true);
			
			const double averageLearningRate = batchData.networkTrainingData.hyperParameters.GetAverageLearningRate();
			const double regularizationFactor = 1.0 - (batchData.networkTrainingData.hyperParameters.learningRate * batchData.networkTrainingData.hyperParameters.lambda) / batchData.networkTrainingData.trainingData.GetNumberOfSamples();
			for (size_t l = 0; l < this->_topology.GetSize(); ++l)
				this->_topology[l]->Update(this->_biasGradients[l], this->_weightGradients[l], averageLearningRate, regularizationFactor);
			
			sw.Stop();
			
			if (batchData.networkTrainingData.debugLevel > 3)
				std::cout << "\t\tUBW[" << batchData.startIndex << ", " << batchData.endIndex << "] completed in " << sw.GetMilliSeconds() << "ms" << std::endl;
		}
		
	
	protected:
		const size_t _miniBatchSize;
		const std::unique_ptr<IShuffler<mathDomain>> _miniBatchShuffler;
	};
}
