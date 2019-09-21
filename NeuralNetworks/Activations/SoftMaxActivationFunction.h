#pragma once

#include <NeuralNetworks/Activations/IActivationFunction.h>

namespace nn
{
	template<MathDomain mathDomain>
	class SoftMaxActivationFunction final: public IActivationFunction<mathDomain>
	{
	public:
		constexpr ActivationFunctionType GetType() const noexcept override { return ActivationFunctionType::SoftMax; }
		
		void Evaluate(typename IActivationFunction<mathDomain>::Matrix& output, const typename IActivationFunction<mathDomain>::Matrix& input) const noexcept override
		{
			auto columnSumCacheIter = _columnSumCache.find(input.nCols());
			if (columnSumCacheIter == _columnSumCache.end())
				columnSumCacheIter = _columnSumCache.emplace(std::piecewise_construct,
				                                   std::forward_as_tuple(input.nCols()),
				                                   std::forward_as_tuple(Vector<mathDomain>(static_cast<unsigned>(input.nCols()), 0.0))).first;
			
			auto onesCacheIter = _onesCache.find(input.nCols());
			if (onesCacheIter == _onesCache.end())
				onesCacheIter = _onesCache.emplace(std::piecewise_construct,
				                                   std::forward_as_tuple(input.nRows()),
				                                   std::forward_as_tuple(Vector<mathDomain>(static_cast<unsigned>(input.nRows()), 1.0))).first;
			
			nn::detail::SoftMax(output.GetBuffer(), input.GetBuffer(), columnSumCacheIter->second.GetBuffer(), onesCacheIter->second.GetBuffer());
		}
		
		void EvaluateGradient(typename IActivationFunction<mathDomain>::Matrix&, const typename IActivationFunction<mathDomain>::Matrix&) const noexcept override
		{
			// doesn't really need to compute the gradient, as this is gonna be used with the cross entropy function only!
		}
		
	private:
		mutable std::unordered_map<size_t, typename IActivationFunction<mathDomain>::Vector> _columnSumCache {};
		mutable std::unordered_map<size_t, typename IActivationFunction<mathDomain>::Vector> _onesCache {};
	};
}
