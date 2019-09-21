#pragma once

#include <NeuralNetworks/Activations/IActivationFunction.h>
#include <NeuralNetworks/NeuralNetworksManager.h>

namespace nn
{
	template<MathDomain mathDomain>
	class SigmoidActivationFunction final: public IActivationFunction<mathDomain>
	{
	public:
		constexpr ActivationFunctionType GetType() const noexcept override { return ActivationFunctionType::Sigmoid; }
		
		void Evaluate(typename IActivationFunction<mathDomain>::Matrix& output, const typename IActivationFunction<mathDomain>::Matrix& input) const noexcept override
		{
			nn::detail::Sigmoid(output.GetBuffer(), input.GetBuffer());
		}
		
		void EvaluateGradient(typename IActivationFunction<mathDomain>::Matrix& output, const typename IActivationFunction<mathDomain>::Matrix& input) const noexcept override
		{
			nn::detail::SigmoidPrime(output.GetBuffer(), input.GetBuffer());
		}
	};
}
