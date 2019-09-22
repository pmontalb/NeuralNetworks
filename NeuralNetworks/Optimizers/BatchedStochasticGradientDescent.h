#pragma once

#include <NeuralNetworks/Optimizers/BatchedGradientOptimizer.h>

namespace nn
{
	template<MathDomain mathDomain>
	class BatchedStochasticGradientDescent final: public BatchedGradientOptimizer<mathDomain>
	{
	public:
		BatchedStochasticGradientDescent(const NetworkTopology<mathDomain>& topology,
		                                 const size_t miniBatchSize,
				                         std::unique_ptr<ICostFunction<mathDomain>>&& costFunction,
				                         std::unique_ptr<IShuffler<mathDomain>>&& miniBatchShuffler) noexcept
			: BatchedGradientOptimizer<mathDomain>(topology, miniBatchSize, std::move(costFunction), std::move(miniBatchShuffler))
		{
			for (const auto& layer: this->_topology)
			{
				_biasGradientCache.emplace_back(
						Matrix<mathDomain>(static_cast<unsigned>(layer->GetBias().size()),
						                   static_cast<unsigned>(this->_miniBatchSize), 0.0));
				
				_weightGradientCache.emplace_back(
						Tensor<mathDomain>(static_cast<unsigned>(layer->GetWeight().nRows()),
						                   static_cast<unsigned>(layer->GetWeight().nCols()),
						                   static_cast<unsigned>(this->_miniBatchSize), 0.0));
			}
		}
		
	private:
		virtual void TrainMiniBatch(MiniBatchData<mathDomain>& batchData) noexcept
		{
			// reset cache
			for (size_t l = 0; l < this->_topology.GetSize(); ++l)
			{
				dm::detail::Zero(this->_biasGradients[l].GetBuffer());
				dm::detail::Zero(this->_weightGradients[l].GetBuffer());
			}
			
			// calculates analytically the gradient, by means of backward differentiation
			_needGradient = this->_topology.back()->GetBestCostFunctionType() != this->_costFunction->GetType();
			AdjointDifferentiation(batchData);
		}
		
		void AdjointDifferentiation(MiniBatchData<mathDomain>& batchData) noexcept
		{
			Stopwatch sw(true);
			const size_t nLayers = this->_topology.GetSize();
			
			// reset weight cache
			for (size_t l = 0; l < nLayers; ++l)
				dm::detail::Zero(this->_weightGradientCache[l].GetBuffer());

			const size_t actualMiniBatchSize = batchData.endIndex - batchData.startIndex;  // last iteration is spurious
			auto onesCacheIter = _onesCache.find(actualMiniBatchSize);
			if (onesCacheIter == _onesCache.end())
				onesCacheIter = _onesCache.emplace(std::piecewise_construct,
						                           std::forward_as_tuple(actualMiniBatchSize),
						                           std::forward_as_tuple(Vector<mathDomain>(static_cast<unsigned>(actualMiniBatchSize), 1.0))).first;
			// network evaluation: feed forward
			Matrix<mathDomain> input(batchData.networkTrainingData.trainingData.input, batchData.startIndex, batchData.endIndex);
			this->_topology.Evaluate(input, _needGradient);
			
			// *** Back propagation of the last layer ***
			// NB override last layer's activation with the cost function derivative!
			Matrix<mathDomain> expectedOutput(batchData.networkTrainingData.trainingData.expectedOutput, batchData.startIndex, batchData.endIndex);
			auto& costFunctionGradient = this->_topology.back()->GetActivation();
			assert(costFunctionGradient.size() == expectedOutput.size());
			this->_costFunction->EvaluateGradient(costFunctionGradient, expectedOutput, this->_topology.back()->GetActivationGradient());
			
			costFunctionGradient.RowWiseSum(this->_biasGradients.back(), onesCacheIter->second);
			Tensor<mathDomain>::KroneckerProduct(this->_weightGradientCache.back(),
			                                     costFunctionGradient,
			                                     this->_topology[nLayers - 2]->GetActivation());
			this->_weightGradientCache.back().CubeWiseSum(this->_weightGradients.back());
			// ***
			
			// now back-propagate through the remaining layers
			for (size_t l = 2; l <= nLayers; ++l)
			{
				auto& nextLayer = this->_topology[nLayers - l + 1];
				auto& layer     = this->_topology[nLayers - l];
				
				nextLayer->GetWeight().Multiply(_biasGradientCache[nLayers - l],
						                        l == 2 ? costFunctionGradient : _biasGradientCache[nLayers - l + 1], MatrixOperation::Transpose);
				
				_biasGradientCache[nLayers - l] %= layer->GetActivationGradient();
				_biasGradientCache[nLayers - l].RowWiseSum(this->_biasGradients[nLayers - l], onesCacheIter->second);
				
				Tensor<mathDomain>::KroneckerProduct(this->_weightGradientCache[nLayers - l],
				                                     _biasGradientCache[nLayers - l],
				                                     l == 2 ? input : this->_topology[nLayers - l - 1]->GetActivation());
				this->_weightGradientCache[nLayers - l].CubeWiseSum(this->_weightGradients[nLayers - l]);
			}
			
			sw.Stop();
			
			if (batchData.networkTrainingData.debugLevel > 3)
				std::cout << "\t\tAD[" << batchData.startIndex << ", " << batchData.endIndex << "] completed in " << sw.GetMilliSeconds() << "ms" << std::endl;
		}
		
	private:
		std::vector<Matrix<mathDomain>> _biasGradientCache;
		std::vector<Tensor<mathDomain>> _weightGradientCache;
		
		std::unordered_map<size_t, Vector<mathDomain>> _onesCache;
		
		bool _needGradient = true;
	};
	
	template<MathDomain mathDomain>
	using BatchedSgd = BatchedStochasticGradientDescent<mathDomain>;
}
