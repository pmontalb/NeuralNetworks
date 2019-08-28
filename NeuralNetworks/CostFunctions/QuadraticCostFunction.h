#pragma once

#include <NeuralNetworks/CostFunctions/CostFunction.h>

namespace nn
{
	template<MathDomain mathDomain>
	class QuadraticCostFunction final: public CostFunction<mathDomain>
	{
	public:
		using CostFunction<mathDomain>::CostFunction;
		
		double EvaluateWorker(typename ICostFunction<mathDomain>::Matrix& expected, const typename ICostFunction<mathDomain>::Matrix& actual) const noexcept override
		{
			expected -= actual;
			double norm2 = expected.EuclideanNorm();
			norm2 *= 0.5 * norm2;

			return norm2;
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
