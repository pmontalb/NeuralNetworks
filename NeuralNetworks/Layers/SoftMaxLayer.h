#pragma once

#include <NeuralNetworks/Layers/DenseLayer.h>
#include <NeuralNetworks/Activations/SoftMaxActivationFunction.h>
#include <NeuralNetworks/CostFunctions/CrossEntropyCostFunctionSoftMax.h>

#include <memory>

namespace nn
{
	// NB: due to its convoluted gradient, this layer must be used only with its cross entropy cost function, and only
	// as a last layer
	template<MathDomain mathDomain>
	class SoftMaxLayer final: public DenseLayer<mathDomain>
	{
	public:
		SoftMaxLayer(const unsigned nInput, const unsigned nOutput,
				     std::unique_ptr<IActivationFunction<mathDomain>>&&,  // blissfully ignored
				     IBiasWeightInitializer<mathDomain>&& initializer)
			: DenseLayer<mathDomain>(nInput, nOutput, std::make_unique<SoftMaxActivationFunction<mathDomain>>(), std::move(initializer))
		{
		}
		
		constexpr LayerType GetType() const noexcept override { return LayerType::SoftMax; }
		
		std::unique_ptr<ICostFunction<mathDomain>> GetCrossEntropyCostFunction() const noexcept override
		{
			return std::make_unique<CrossEntropyCostFunctionSoftMax<mathDomain>>();
		}
	};
}
