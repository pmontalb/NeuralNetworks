#pragma once

#include <NeuralNetworks/Activations/IActivationFunction.h>

namespace nn
{
	template<MathDomain mathDomain>
	class InverseSquareRootLinearUnitActivationFunction final: public IActivationFunction<mathDomain>
	{
	public:
		void Evaluate(typename IActivationFunction<mathDomain>::Vector& output, const typename IActivationFunction<mathDomain>::Vector& input) const noexcept override
		{
			nn::detail::InverseSquareRootLinearUnit(output.GetBuffer(), input.GetBuffer());
		}
		
		void EvaluateGradient(typename IActivationFunction<mathDomain>::Vector& output, const typename IActivationFunction<mathDomain>::Vector& input) const noexcept override
		{
			nn::detail::InverseSquareRootLinearUnitPrime(output.GetBuffer(), input.GetBuffer());
		}
	};
	
	template<MathDomain mathDomain>
	using IsrLuActivationFunction = RectifiedLinearUnitActivationFunction<mathDomain>;
}
