#pragma once

#include <NeuralNetworks/Initializers/RandomBiasWeightInitializer.h>

namespace nn
{
	template<MathDomain mathDomain>
	class SmallVarianceRandomBiasWeightInitializer final: public RandomBiasWeightInitializer<mathDomain>
	{
	public:
		using RandomBiasWeightInitializer<mathDomain>::RandomBiasWeightInitializer;
		
		void Set(typename IBiasWeightInitializer<mathDomain>::Weight& weight) const noexcept override
		{
			RandomBiasWeightInitializer<mathDomain>::Set(weight);
			weight.Scale(1.0 / std::sqrt(weight.nCols()));
		}
	};
}
