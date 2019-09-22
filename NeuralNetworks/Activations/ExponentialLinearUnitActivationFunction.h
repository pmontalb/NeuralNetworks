#pragma once

#include <NeuralNetworks/Activations/IActivationFunction.h>

namespace nn
{
	template<MathDomain mathDomain>
	class ExponentialLinearUnitActivationFunction final: public IActivationFunction<mathDomain>
	{
	public:
		constexpr ActivationFunctionType GetType() const noexcept override { return ActivationFunctionType::ExponentialLinearUnity; }
		constexpr CostFunctionType GetBestCostFunction() const noexcept override { return CostFunctionType::Null; }
		
		void Evaluate(typename IActivationFunction<mathDomain>::Matrix& output, const typename IActivationFunction<mathDomain>::Matrix& input) const noexcept override
		{
			nn::detail::ExponentialLinearUnit(output.GetBuffer(), input.GetBuffer());
		}
		
		void EvaluateGradient(typename IActivationFunction<mathDomain>::Matrix& output, const typename IActivationFunction<mathDomain>::Matrix& input, const typename IActivationFunction<mathDomain>::Matrix&) const noexcept override
		{
			nn::detail::ExponentialLinearUnitPrime(output.GetBuffer(), input.GetBuffer());
		}
	};
	
	template<MathDomain mathDomain>
	using EluActivationFunction = RectifiedLinearUnitActivationFunction<mathDomain>;
}
