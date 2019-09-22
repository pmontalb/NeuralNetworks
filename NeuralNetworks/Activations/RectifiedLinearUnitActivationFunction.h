#pragma once

#include <NeuralNetworks/Activations/IActivationFunction.h>

namespace nn
{
	template<MathDomain mathDomain>
	class RectifiedLinearUnitActivationFunction final: public IActivationFunction<mathDomain>
	{
	public:
		constexpr ActivationFunctionType GetType() const noexcept override { return ActivationFunctionType::RectifiedLinearUnit; }
		constexpr CostFunctionType GetBestCostFunction() const noexcept override { return CostFunctionType::Null; }
		
		void Evaluate(typename IActivationFunction<mathDomain>::Matrix& output, const typename IActivationFunction<mathDomain>::Matrix& input) const noexcept override
		{
			nn::detail::RectifiedLinearUnit(output.GetBuffer(), input.GetBuffer());
		}
		
		void EvaluateGradient(typename IActivationFunction<mathDomain>::Matrix& output, const typename IActivationFunction<mathDomain>::Matrix& input, const typename IActivationFunction<mathDomain>::Matrix&) const noexcept override
		{
			nn::detail::RectifiedLinearUnitPrime(output.GetBuffer(), input.GetBuffer());
		}
	};
	
	template<MathDomain mathDomain>
	using ReLuActivationFunction = RectifiedLinearUnitActivationFunction<mathDomain>;
}
