#pragma once

#include <NeuralNetworks/Layers/Initializers/IBiasWeightInitializer.h>

namespace nn
{
	template<MathDomain mathDomain>
	class TrivialBiasWeightInitializer final: public IBiasWeightInitializer<mathDomain>
	{
	public:
		using IBiasWeightInitializer<mathDomain>::IBiasWeightInitializer;
		
		void Set(typename IBiasWeightInitializer<mathDomain>::Weight&) const noexcept override
		{
		}
		
		void Set(typename IBiasWeightInitializer<mathDomain>::Bias&) const noexcept override
		{
		}
	};
}
