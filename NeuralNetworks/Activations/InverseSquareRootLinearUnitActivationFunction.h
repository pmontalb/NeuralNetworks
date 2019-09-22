#pragma once

#include <NeuralNetworks/Activations/IActivationFunction.h>

namespace nn
{
	template<MathDomain mathDomain>
	class InverseSquareRootLinearUnitActivationFunction final: public IActivationFunction<mathDomain>
	{
	public:
		constexpr ActivationFunctionType GetType() const noexcept override { return ActivationFunctionType::InverseSquareRootLinearUnit; }
		constexpr CostFunctionType GetBestCostFunction() const noexcept override { return CostFunctionType::Null; }
		
		void Evaluate(typename IActivationFunction<mathDomain>::Matrix& output, const typename IActivationFunction<mathDomain>::Matrix& input) const noexcept override
		{
			nn::detail::InverseSquareRootLinearUnit(output.GetBuffer(), input.GetBuffer());
		}
		
		void EvaluateGradient(typename IActivationFunction<mathDomain>::Matrix& output, const typename IActivationFunction<mathDomain>::Matrix& input, const typename IActivationFunction<mathDomain>::Matrix&) const noexcept override
		{
			nn::detail::InverseSquareRootLinearUnitPrime(output.GetBuffer(), input.GetBuffer());
		}
	};
	
	template<MathDomain mathDomain>
	using IsrLuActivationFunction = RectifiedLinearUnitActivationFunction<mathDomain>;
}
