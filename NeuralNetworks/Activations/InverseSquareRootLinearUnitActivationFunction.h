#pragma once

#include <NeuralNetworks/Activations/IActivationFunction.h>

namespace nn
{
	template<MathDomain mathDomain>
	class LeakyRectifiedLinearUnitActivationFunction final: public IActivationFunction<mathDomain>
	{
	public:
		void Evaluate(typename IActivationFunction<mathDomain>::Vector& output, const typename IActivationFunction<mathDomain>::Vector& input) const noexcept override
		{
			nn::detail::LeakyRectifiedLinearUnit(output.GetBuffer(), input.GetBuffer());
		}
		
		void EvaluateGradient(typename IActivationFunction<mathDomain>::Vector& output, const typename IActivationFunction<mathDomain>::Vector& input) const noexcept override
		{
			nn::detail::LeakyRectifiedLinearUnitPrime(output.GetBuffer(), input.GetBuffer());
		}
	};
	
	template<MathDomain mathDomain>
	using ReLuActivationFunction = RectifiedLinearUnitActivationFunction<mathDomain>;
}