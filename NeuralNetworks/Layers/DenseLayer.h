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
			this->_zVector += this->_bias;
			
			// if output is not provided, use the activation buffers, and compute the gradient as well!
			if (!output)
			{
				this->_activationFunction->Evaluate(this->_activation, this->_zVector);
				this->_activationFunction->EvaluateGradient(this->_activationGradient, this->_zVector);
			}
			else
				this->_activationFunction->Evaluate(*output, this->_zVector);
		}
	};
}
