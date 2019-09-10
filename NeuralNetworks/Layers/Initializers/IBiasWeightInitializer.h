#pragma once

#include <Types.h>

namespace nn
{
	template<MathDomain mathDomain>
	class IBiasWeightInitializer
	{
	public:
		using Weight = cl::ColumnWiseMatrix<MemorySpace::Device, mathDomain>;
		using Bias = cl::Vector<MemorySpace::Device, mathDomain>;
		
		virtual ~IBiasWeightInitializer() = default;
		virtual void Set(Weight& weight) const noexcept = 0;
		virtual void Set(Bias& bias) const noexcept = 0;
	};
}
