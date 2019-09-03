#pragma once

#include <NeuralNetworks/Activations/IActivationFunction.h>

namespace nn
{
	template<MathDomain mathDomain>
	class ExponentialLinearUnitActivationFunction final: public IActivationFunction<mathDomain>
	{
	public:
		void Evaluate(typename IActivationFunction<mathDomain>::Vector& output, const typename IActivationFunction<mathDomain>::Vector& input) const noexcept override
		{
			nn::detail::ExponentialLinearUnit(output.GetBuffer(), input.GetBuffer());
		}
		
		void EvaluateGradient(typename IActivationFunction<mathDomain>::Vector& output, const typename IActivationFunction<mathDomain>::Vector& input) const noexcept override
		{
			nn::detail::ExponentialLinearUnitPrime(output.GetBuffer(), input.GetBuffer());
		}
	};
	
	template<MathDomain mathDomain>
	using EluActivationFunction = RectifiedLinearUnitActivationFunction<mathDomain>;
}
