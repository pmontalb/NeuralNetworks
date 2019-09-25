#pragma once

#include <NeuralNetworks/Optimizers/BatchedGradientOptimizer.h>

namespace nn
{
	namespace detail
	{
		template<MathDomain md>
		using CsrMatrix = cl::CompressedSparseRowMatrix<MemorySpace::Device, md>;
		
		template<MathDomain mathDomain>
		struct MiniBatchCache
		{
			std::vector<Matrix<mathDomain>> biasGradients;
			std::vector<Tensor<mathDomain>> weightGradients;
			
			Vector<mathDomain> ones;
			
			#ifndef RUN_MULTIPLE_ADD
				std::vector<CsrMatrix<mathDomain>> eye;
			#endif
			
			explicit MiniBatchCache(const NetworkTopology<mathDomain>& topology, const size_t miniBatchSize)
				: ones(static_cast<unsigned>(miniBatchSize), 1.0)
			{
				biasGradients.reserve(topology.GetSize());
				weightGradients.reserve(topology.GetSize());
				
				#ifndef RUN_MULTIPLE_ADD
					eye.reserve(topology.GetSize());
				#endif
				
				for (const auto& layer: topology)
				{
					biasGradients.emplace_back(Matrix<mathDomain>(static_cast<unsigned>(layer->GetBias().size()),
							                                      static_cast<unsigned>(miniBatchSize),
							                                      0.0));
					
					weightGradients.emplace_back(Tensor<mathDomain>(static_cast<unsigned>(layer->GetWeight().nRows()),
							                                        static_cast<unsigned>(layer->GetWeight().nCols()),
							                                        static_cast<unsigned>(miniBatchSize), 0.0));
					
					#ifndef RUN_MULTIPLE_ADD
						const size_t weightMatrixSize = layer->GetWeight().size();
						Vector<MathDomain::Int> nNonZeroRows(static_cast<unsigned>(weightMatrixSize * miniBatchSize));
						nNonZeroRows.LinSpace(0, static_cast<int>(nNonZeroRows.size() - 1));
						nNonZeroRows.Scale(miniBatchSize);
						
						std::vector<int> nonZeroColumnIndicesCpu(nNonZeroRows.size());
						for (size_t i = 0; i < weightMatrixSize; ++i)
						{
							for (size_t k = 0; k < miniBatchSize; ++k)
								nonZeroColumnIndicesCpu[k + miniBatchSize * i] = static_cast<int>(k * weightMatrixSize + i);
						}
						Vector<MathDomain::Int> nonZeroColumnIndices(nonZeroColumnIndicesCpu);
						eye.emplace_back(CsrMatrix<mathDomain>(static_cast<unsigned>(weightMatrixSize),
								                               static_cast<unsigned>(nNonZeroRows.size()),
								                               std::move(nonZeroColumnIndices),
								                               std::move(nNonZeroRows),
								                               1.0));
					#endif
				}
			}
		
			void Reset()
			{
				for (auto& weightGradient: weightGradients)
					dm::detail::Zero(weightGradient.GetBuffer());
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
			
			const size_t actualMiniBatchSize = batchData.endIndex - batchData.startIndex;  // last iteration is spurious
			
			// retrieve cached quantities
			auto cacheIter = _cache.find(actualMiniBatchSize);
			if (cacheIter == _cache.end())
				cacheIter = _cache.emplace(std::piecewise_construct,
				                           std::forward_as_tuple(actualMiniBatchSize),
				                           std::forward_as_tuple(this->_topology, actualMiniBatchSize)).first;
			// reset weight cache
			cacheIter->second.Reset();
			
			// network evaluation: feed forward
			Matrix<mathDomain> input(batchData.networkTrainingData.trainingData.input, batchData.startIndex, batchData.endIndex);
			this->_topology.Evaluate(input, _needGradient);
			
			// *** Back propagation of the last layer ***
			// NB override last layer's activation with the cost function derivative!
			Matrix<mathDomain> expectedOutput(batchData.networkTrainingData.trainingData.expectedOutput, batchData.startIndex, batchData.endIndex);
			auto& costFunctionGradient = this->_topology.back()->GetActivation();
			assert(costFunctionGradient.size() == expectedOutput.size());
			this->_costFunction->EvaluateGradient(costFunctionGradient, expectedOutput, this->_topology.back()->GetActivationGradient());
			
			costFunctionGradient.RowWiseSum(this->_biasGradients.back(), cacheIter->second.ones);
			Tensor<mathDomain>::KroneckerProduct(cacheIter->second.weightGradients.back(),
			                                     costFunctionGradient,
			                                     this->_topology[nLayers - 2]->GetActivation());
			
			#ifndef RUN_MULTIPLE_ADD
				cacheIter->second.weightGradients.back().CubeWiseSum(this->_weightGradients.back(), cacheIter->second.eye.back());
			#else
				cacheIter->second.weightGradients.back().CubeWiseSum(this->_weightGradients.back());
			#endif
			// ***
			
			// now back-propagate through the remaining layers
			for (size_t l = 2; l <= nLayers; ++l)
			{
				auto& nextLayer = this->_topology[nLayers - l + 1];
				auto& layer     = this->_topology[nLayers - l];
				
				nextLayer->GetWeight().Multiply(cacheIter->second.biasGradients[nLayers - l],
						                        l == 2 ? costFunctionGradient : cacheIter->second.biasGradients[nLayers - l + 1], MatrixOperation::Transpose);
				
				cacheIter->second.biasGradients[nLayers - l] %= layer->GetActivationGradient();
				cacheIter->second.biasGradients[nLayers - l].RowWiseSum(this->_biasGradients[nLayers - l], cacheIter->second.ones);
				
				Tensor<mathDomain>::KroneckerProduct(cacheIter->second.weightGradients[nLayers - l],
				                                     cacheIter->second.biasGradients[nLayers - l],
				                                     l == 2 ? input : this->_topology[nLayers - l - 1]->GetActivation());
				
				#ifndef RUN_MULTIPLE_ADD
					cacheIter->second.weightGradients[nLayers - l].CubeWiseSum(this->_weightGradients[nLayers - l], cacheIter->second.eye[nLayers - l]);
				#else
					cacheIter->second.weightGradients[nLayers - l].CubeWiseSum(this->_weightGradients[nLayers - l]);
				#endif
			}
			
			sw.Stop();
			
			if (batchData.networkTrainingData.debugLevel > 3)
				std::cout << "\t\tAD[" << batchData.startIndex << ", " << batchData.endIndex << "] completed in " << sw.GetMilliSeconds() << "ms" << std::endl;
		}
		
	private:
		detail::CacheMap<mathDomain> _cache;
		
		bool _needGradient = true;
	};
	
	template<MathDomain mathDomain>
	using BatchedSgd = BatchedStochasticGradientDescent<mathDomain>;
}
