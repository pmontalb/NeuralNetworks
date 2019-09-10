#pragma once

#include <NeuralNetworks/Layers/Layer.h>
#include <NeuralNetworks/Activations/SoftMaxActivationFunction.h>
#include <NeuralNetworks/CostFunctions/CrossEntropyCostFunctionSoftMax.h>

#include <memory>

namespace nn
{
	// NB: due to its convoluted gradient, this layer must be used only with its cross entropy cost function, and only
	// as a last layer
	template<MathDomain mathDomain>
	class SoftMaxLayer final: public Layer<mathDomain>
	{
	public:
		SoftMaxLayer(const unsigned nInput, const unsigned nOutput,
				     std::unique_ptr<IActivationFunction<mathDomain>>&&,  // blissfully ignored
				     IBiasWeightInitializer<mathDomain>&& initializer)
			: Layer<mathDomain>(nInput, nOutput, std::make_unique<SoftMaxActivationFunction<mathDomain>>(), std::move(initializer))
		{
		}
		
		constexpr LayerType GetType() const noexcept override { return LayerType::SoftMax; }
		
		void Evaluate(const typename Layer<mathDomain>::Vector& input, typename Layer<mathDomain>::Vector* const output) noexcept override
		{
			this->_weight.Dot(this->_zVector, input);
			this->_zVector.AddEqual(this->_bias);
			
			// if output is not provided, use the activation buffers, and compute the gradient as well!
			if (!output)
			{
				this->_activationFunction->Evaluate(this->_activation, this->_zVector);
				// not really needed!
				//this->_activationFunction->EvaluateGradient(this->_activationGradient, this->_zVector);
			}
			else
				this->_activationFunction->Evaluate(*output, this->_zVector);
		}
		
		std::unique_ptr<ICostFunction<mathDomain>> GetCrossEntropyCostFunction() const noexcept override
		{
			return std::make_unique<CrossEntropyCostFunctionSoftMax<mathDomain>>();
		}
	};
}
