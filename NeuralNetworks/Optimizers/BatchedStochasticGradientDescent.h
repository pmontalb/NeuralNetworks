#pragma once

#include <NeuralNetworks/Optimizers/BatchedOptimizer.h>

namespace nn
{
	template<MathDomain mathDomain>
	class BatchedStochasticGradientDescent final: public BatchedOptimizer<mathDomain>
	{
	public:
		BatchedStochasticGradientDescent(const typename BatchedOptimizer<mathDomain>::Layers& layers,
				                         std::unique_ptr<ICostFunction<mathDomain>>&& costFunction,
				                         std::unique_ptr<IShuffler<mathDomain>>&& miniBatchShuffler) noexcept
			: BatchedOptimizer<mathDomain>(layers, std::move(costFunction), std::move(miniBatchShuffler))
		{
			for (size_t l = 0; l < layers.size(); ++l)
				_biasGradientCache.emplace_back(Vector<mathDomain>(static_cast<unsigned>(layers[l]->GetNumberOfOutputs()), 0.0));
		}
		
	private:
		virtual void TrainMiniBatch(MiniBatchData<mathDomain>& batchData) noexcept
		{
			// reset cache
			for (size_t l = 0; l < this->_layers.size(); ++l)
			{
				this->_biasGradients[l].Set(0.0);
				this->_weightGradients[l].Set(0.0);
			}
			
			// calculates analytically the gradient, by means of backward differentiation
			AdjointDifferentiation(batchData);
		}
		
		void AdjointDifferentiation(MiniBatchData<mathDomain>& batchData) noexcept
		{
			Stopwatch sw(true);
			
			const size_t nLayers = this->_layers.size();
			for (size_t i = batchData.startIndex; i < batchData.endIndex; ++i)
			{
				// network evaluation
				this->_layers[0]->Evaluate(*batchData.networkTrainingData.trainingData.input.columns[i]);
				for (size_t l = 1; l < nLayers; ++l)
					this->_layers[l]->Evaluate(this->_layers[l - 1]->GetActivation());
				
				// *** Back propagation of the last layer ***
				// NB override last layer's activation with the cost function derivative!
				auto& costFunctionGradient = this->_layers.back()->GetActivation();
				assert(costFunctionGradient.size() == batchData.networkTrainingData.trainingData.expectedOutput.columns[i]->size());
				this->_costFunction->EvaluateGradient(costFunctionGradient,
						                              *batchData.networkTrainingData.trainingData.expectedOutput.columns[i],
						                              this->_layers.back()->GetActivationGradient());
				
				// back propagation
				this->_biasGradients.back().AddEqual(costFunctionGradient);
				Matrix<mathDomain>::KroneckerProduct(this->_weightGradients.back(),
				                                     costFunctionGradient,
				                                     this->_layers[nLayers - 2]->GetActivation());
				// ***
				
				// now back-propagate through the remaining layers
				for (size_t l = 2; l <= nLayers; ++l)
				{
					auto& nextLayer = this->_layers[nLayers - l + 1];
					auto& layer     = this->_layers[nLayers - l];
					
					nextLayer->GetWeight().Dot(_biasGradientCache[nLayers - l],
					                           l == 2 ? costFunctionGradient : _biasGradientCache[nLayers - l + 1],
					                           MatrixOperation::Transpose);
					
					_biasGradientCache[nLayers - l] %= layer->GetActivationGradient();
					this->_biasGradients[nLayers - l] += _biasGradientCache[nLayers - l];
					Matrix<mathDomain>::KroneckerProduct(this->_weightGradients[nLayers - l],
					                                     _biasGradientCache[nLayers - l],
					                                     l == 2 ? *batchData.networkTrainingData.trainingData.input.columns[i] : this->_layers[nLayers - l - 1]->GetActivation());
				}
			}
			
			sw.Stop();
			
			if (batchData.networkTrainingData.debugLevel > 3)
				std::cout << "\t\tAD[" << batchData.startIndex << ", " << batchData.endIndex << "] completed in " << sw.GetMilliSeconds() << "ms" << std::endl;
		}
		
	private:
		std::vector<Vector<mathDomain>> _biasGradientCache;
	};
	
	template<MathDomain mathDomain>
	using BatchedSgd = BatchedStochasticGradientDescent<mathDomain>;
}
