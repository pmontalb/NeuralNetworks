#pragma once

#include <NeuralNetworks/Layers/Layer.h>

namespace nn
{
	template<MathDomain mathDomain>
	class DenseLayer: public Layer<mathDomain>
	{
	public:
		DenseLayer(const unsigned nInput,
		      const unsigned nOutput,
		      std::unique_ptr<IActivationFunction<mathDomain>>&& activationFunction,
		      IBiasWeightInitializer<mathDomain>&& initializer)
				: Layer<mathDomain>(nInput,nOutput, std::move(activationFunction), std::move(initializer))
		{
		}
		
		constexpr LayerType GetType() const noexcept override { return LayerType::Dense; }
		
		void Evaluate(const typename Layer<mathDomain>::Matrix& input, typename Layer<mathDomain>::Matrix* const output) noexcept override
		{
			auto zMatrixIter = this->_zMatrix.find(input.nCols());
			if (zMatrixIter == this->_zMatrix.end())
				zMatrixIter = this->_zMatrix.emplace(std::piecewise_construct,
				                                     std::forward_as_tuple(input.nCols()),
				                                     std::forward_as_tuple(typename Layer<mathDomain>::Matrix(this->_weight.nRows(), input.nCols()))).first;
			
			this->_weight.Multiply(zMatrixIter->second, input);
			
			auto onesCacheIter = _onesCache.find(input.nCols());
			if (onesCacheIter == _onesCache.end())
				onesCacheIter = _onesCache.emplace(std::piecewise_construct,
						                           std::forward_as_tuple(input.nCols()),
				                                   std::forward_as_tuple(typename Layer<mathDomain>::Vector(input.nCols(), 1.0))).first;
			zMatrixIter->second.AddEqual(this->_bias, onesCacheIter->second, false);
			
			// if output is not provided, use the activation buffers, and compute the gradient as well!
			if (!output)
			{
				auto activationIter = this->_batchedActivation.find(input.nCols());
				if (activationIter == this->_batchedActivation.end())
					activationIter = this->_batchedActivation.emplace(std::piecewise_construct,
					                                                  std::forward_as_tuple(input.nCols()),
					                                                  std::forward_as_tuple(typename Layer<mathDomain>::Matrix(zMatrixIter->second.nRows(), zMatrixIter->second.nCols()))).first;
				this->_lastActivation = &activationIter->second;
				
				auto activationGradientIter = this->_batchedActivationGradient.find(input.nCols());
				if (activationGradientIter == this->_batchedActivationGradient.end())
					activationGradientIter = this->_batchedActivationGradient.emplace(std::piecewise_construct,
					                                                                  std::forward_as_tuple(input.nCols()),
					                                                                  std::forward_as_tuple(typename Layer<mathDomain>::Matrix(zMatrixIter->second.nRows(), zMatrixIter->second.nCols()))).first;
				this->_lastActivationGradient = &activationGradientIter->second;
				
				assert(activationIter->second.size() == zMatrixIter->second.size());
				this->_activationFunction->Evaluate(activationIter->second, zMatrixIter->second);
				this->_activationFunction->EvaluateGradient(activationGradientIter->second, zMatrixIter->second);
			}
			else
				this->_activationFunction->Evaluate(*output, zMatrixIter->second);
		}
		
	private:
		std::unordered_map<size_t, typename ILayer<mathDomain>::Vector> _onesCache {};
	};
}
