#pragma once

#include <NeuralNetworks/Layers/Initializers/IBiasWeightInitializer.h>

namespace nn
{
	template<MathDomain mathDomain>
	class RandomBiasWeightInitializer: public IBiasWeightInitializer<mathDomain>
	{
	public:
		using IBiasWeightInitializer<mathDomain>::IBiasWeightInitializer;
		
		void Set(typename IBiasWeightInitializer<mathDomain>::Weight& weight) const noexcept override
		{
			weight.RandomGaussian();
		}
		
		void Set(typename IBiasWeightInitializer<mathDomain>::Bias& bias) const noexcept override final
		{
			bias.RandomGaussian();
		}
	};
}
