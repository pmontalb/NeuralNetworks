#pragma once

#include <NeuralNetworks/Optimizers/BatchedGradientOptimizer.h>
#include <TensorCollection.h>

namespace nn
{
	namespace detail
	{
		template<MathDomain mathDomain>
		static inline std::vector<std::tuple<size_t, size_t, size_t>> GetWeightSizes(const NetworkTopology<mathDomain> &topology, const size_t miniBatchSize)
		{
			std::vector<std::tuple<size_t, size_t, size_t>> ret;
			auto sizes = topology.GetTransposedSizes();
			for (auto& size: sizes)
				ret.emplace_back(std::make_tuple(size.first, size.second, miniBatchSize));
			
			return ret;
		}
		
		template<MathDomain mathDomain>
		static inline std::vector<std::pair<size_t, size_t>> GetBiasSizes(const NetworkTopology<mathDomain> &topology, const size_t miniBatchSize)
		{
			std::vector<std::pair<size_t, size_t>> ret;
			auto sizes = topology.GetTransposedSizes();
			for (auto& size: sizes)
				ret.emplace_back(size.first, miniBatchSize);
			
			return ret;
		}
		
		template<MathDomain mathDomain>
		struct MiniBatchCache
		{
			cl::ColumnWiseMatrixCollection<MemorySpace::Device, mathDomain> biasGradients;
			cl::TensorCollection<MemorySpace::Device, mathDomain> weightGradients;
			Vector<mathDomain> ones;
			
			explicit MiniBatchCache(const NetworkTopology<mathDomain>& topology, const size_t miniBatchSize)
				: biasGradients(GetBiasSizes(topology, miniBatchSize)),
				  weightGradients(GetWeightSizes(topology, miniBatchSize)),
				  ones(static_cast<unsigned>(miniBatchSize), 1.0)
			{
			}
		
			void Reset()
			{
				dm::detail::Zero(weightGradients.Get().GetBuffer());
			}
		};
		
		template<MathDomain mathDomain>
		using CacheMap = std::unordered_map<size_t, MiniBatchCache<mathDomain>>;
	}
	
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
		}
		
	private:
		virtual void TrainMiniBatch(MiniBatchData<mathDomain>& batchData) noexcept override
		{
			// reset cache
			dm::detail::Zero(this->_biasGradients.Get().GetBuffer());
			dm::detail::Zero(this->_weightGradients.Get().GetBuffer());
			
			// calculates analytically the gradient, by means of backward differentiation
			_needGradient = this->_topology.back()->GetBestCostFunctionType() != this->_costFunction->GetType();
			AdjointDifferentiation(batchData);
		}
		
		void AdjointDifferentiation(MiniBatchData<mathDomain>& batchData) noexcept
		{
			Stopwatch sw(true);
			const size_t nLayers = this->_topology.GetSize();
			
			const size_t actualMiniBatchSize = batchData.endIndex - batchData.startIndex;  // last iteration is spurious
			
			// retrieve cached quantities
			auto cacheIter = _cache.find(actualMiniBatchSize);
			if (cacheIter == _cache.end())
				cacheIter = _cache.emplace(std::piecewise_construct, std::forward_as_tuple(actualMiniBatchSize),
				                           std::forward_as_tuple(this->_topology, actualMiniBatchSize)).first;
			else
				// reset weight gradient cache
				cacheIter->second.Reset();
			auto& cache = cacheIter->second;
			
			// network evaluation: feed forward
			Matrix<mathDomain> input(batchData.networkTrainingData.trainingData.input, batchData.startIndex, batchData.endIndex);
			this->_topology.Evaluate(input, _needGradient);  // compute y = f(z_L)
			
			// *** Back propagation of the last layer ***
			Matrix<mathDomain> expectedOutput(batchData.networkTrainingData.trainingData.expectedOutput, batchData.startIndex, batchData.endIndex);
			auto& costFunctionGradient = this->_topology.back()->GetActivation();  // dL/dy \outerdot f'(z_L) (delta_L in some literature)
			// NB override last layer's activation with the cost function derivative!
			this->_costFunction->EvaluateGradient(costFunctionGradient, expectedOutput, this->_topology.back()->GetActivationGradient());
			costFunctionGradient.RowWiseSum(this->_biasGradients.back(), cache.ones);  // dL/db_L == dL/dy
			
			// dL/dW_L = dL/db_L \cdot f(z_{L - 1})
			Tensor<mathDomain>::AccumulateKroneckerProduct(this->_weightGradients.back(),
					                                       costFunctionGradient,
					                                       this->_topology[nLayers - 2]->GetActivation());
			//***
			
			// now back-propagate through the remaining layers
			for (size_t l = 2; l <= nLayers; ++l)
			{
				// dL/db_l = (W_l^T * dL/db_{l + 1}) \outerdot f'(z_l)
				this->_topology[nLayers - l + 1]->GetWeight().Multiply(cache.biasGradients[nLayers - l],
						                                               l == 2 ? costFunctionGradient : cache.biasGradients[nLayers - l + 1], MatrixOperation::Transpose);
				cache.biasGradients[nLayers - l] %= this->_topology[nLayers - l]->GetActivationGradient();
				cache.biasGradients[nLayers - l].RowWiseSum(this->_biasGradients[nLayers - l], cache.ones);
				
				// dL/dW_l = dL/db_l \cdot f(z_{L - 1})
				Tensor<mathDomain>::AccumulateKroneckerProduct(this->_weightGradients[nLayers - l],
				                                               cache.biasGradients[nLayers - l],
				                                               l == 2 ? input : this->_topology[nLayers - l - 1]->GetActivation());
			}
			
			sw.Stop();
			
			if (batchData.networkTrainingData.debugLevel > 3)
				std::cout << "\t\tAD[" << batchData.startIndex << ", " << batchData.endIndex << "] completed in " << sw.GetMilliSeconds() << "ms" << std::endl;
		}
		
	private:
		detail::CacheMap<mathDomain> _cache {};
		
		bool _needGradient = true;
	};
	
	template<MathDomain mathDomain>
	using BatchedSgd = BatchedStochasticGradientDescent<mathDomain>;
}
