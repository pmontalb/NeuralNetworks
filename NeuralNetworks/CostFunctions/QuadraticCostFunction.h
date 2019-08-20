#pragma once

namespace nn
{
	template<MathDomain mathDomain>
	class QuadraticCostFunction final: public ICostFunction<mathDomain>
	{
	public:
		using ICostFunction<mathDomain>::ICostFunction;
		
		void Evaluate(typename ICostFunction<mathDomain>::Vector& expected, const typename ICostFunction<mathDomain>::Vector& actual) const noexcept override
		{
			expected -= actual;
			// TODO: sum squares
			expected.Scale(0.5);
		}
		
		void EvaluateDerivative(typename ICostFunction<mathDomain>::Vector& expected,
							    const typename ICostFunction<mathDomain>::Vector& actual,
							    const typename ICostFunction<mathDomain>::Vector& activationDerivative) const noexcept override
		{
			expected -= actual;
			expected %= activationDerivative;
		}
	};
}