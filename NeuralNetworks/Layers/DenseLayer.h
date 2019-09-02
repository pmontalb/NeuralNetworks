#pragma once

#include <NeuralNetworks/Layers/Layer.h>

namespace nn
{
	template<MathDomain mathDomain>
	class DenseLayer final: public Layer<mathDomain>
	{
	public:
		using Layer<mathDomain>::Layer;
		
		void Evaluate(const typename Layer<mathDomain>::Vector& input, typename Layer<mathDomain>::Vector* const output) noexcept override
		{
			this->_weight.Dot(this->_zVector, input);
			this->_zVector.AddEqual(this->_bias);
			
			// if output is not provided, use the activation buffers, and compute the gradient as well!
			if (!output)
			{
				nn::detail::Sigmoid(this->_activation.GetBuffer(), this->_zVector.GetBuffer());
				nn::detail::SigmoidPrime(this->_activationGradient.GetBuffer(), this->_zVector.GetBuffer());
			}
			else
				nn::detail::Sigmoid(output->GetBuffer(), this->_zVector.GetBuffer());
		}
	};
}