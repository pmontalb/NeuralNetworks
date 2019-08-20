#pragma once

#include <Types.h>

namespace nn
{
	template<MathDomain mathDomain>
	class ICostFunction
	{
	public:
		using Vector = cl::Vector<MemorySpace::Device, mathDomain>;
		
		virtual ~ICostFunction() = default;
		virtual void Evaluate(Vector& expected, const Vector& actual) const noexcept = 0;
		virtual void EvaluateDerivative(Vector& expected, const Vector& actual, const Vector& activationDerivative) const noexcept = 0;
	};
}
