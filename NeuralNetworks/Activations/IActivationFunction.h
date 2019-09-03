#pragma once

#include <Types.h>

namespace nn
{
	template<MathDomain mathDomain>
	class IActivationFunction
	{
	public:
		using Vector = cl::Vector<MemorySpace::Device, mathDomain>;
		
		virtual ~IActivationFunction() = default;
		virtual void Evaluate(Vector& output, const Vector& input) const noexcept = 0;
		virtual void EvaluateGradient(Vector& output, const Vector& input) const noexcept = 0;
	};
}
