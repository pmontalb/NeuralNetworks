#pragma once

#include <NeuralNetworks/Initializers/IBiasWeightInitializer.h>

namespace nn
{
	template<MathDomain mathDomain>
	class ZeroBiasWeightInitializer final: public IBiasWeightInitializer<mathDomain>
	{
	public:
		using IBiasWeightInitializer<mathDomain>::IBiasWeightInitializer;
		
		void Set(typename IBiasWeightInitializer<mathDomain>::Weight& weight) const noexcept override
		{
			weight.Set(0.0);
		}
		
		void Set(typename IBiasWeightInitializer<mathDomain>::Bias& bias) const noexcept override
		{
			bias.Set(0.0);
		}
	};
}
