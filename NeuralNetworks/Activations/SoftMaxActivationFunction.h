#pragma once

#include <NeuralNetworks/Activations/IActivationFunction.h>

namespace nn
{
	template<MathDomain mathDomain>
	class SoftMaxActivationFunction final: public IActivationFunction<mathDomain>
	{
	public:
		void Evaluate(typename IActivationFunction<mathDomain>::Vector& output, const typename IActivationFunction<mathDomain>::Vector& input) const noexcept override
		{
			nn::detail::SoftMax(output.GetBuffer(), input.GetBuffer());
		}
		
		void EvaluateGradient(typename IActivationFunction<mathDomain>::Vector&, const typename IActivationFunction<mathDomain>::Vector&) const noexcept override
		{
			// TODO: not computing it for now!
			// doesn't really need to compute the gradient, as this is gonna be used with the cross entropy function only!
		}
	};
}
