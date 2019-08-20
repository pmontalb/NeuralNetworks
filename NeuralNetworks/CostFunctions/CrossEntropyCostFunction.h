#pragma once

namespace nn
{
	template<MathDomain mathDomain>
	class CrossEntropyCostFunction final: public ICostFunction<mathDomain>
	{
	public:
		using ICostFunction<mathDomain>::ICostFunction;
		
		void Evaluate(typename ICostFunction<mathDomain>::Vector&, const typename ICostFunction<mathDomain>::Vector&) const noexcept override
		{
			// TODO: write cross entropy kernel
		}
		
		void EvaluateDerivative(typename ICostFunction<mathDomain>::Vector& expected,
								const typename ICostFunction<mathDomain>::Vector& actual,
								const typename ICostFunction<mathDomain>::Vector& activationDerivative) const noexcept override
		{
			expected -= actual;
		}
	};
}