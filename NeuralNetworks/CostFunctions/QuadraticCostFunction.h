#pragma once

#include <NeuralNetworks/CostFunctions/CostFunction.h>

namespace nn
{
	template<MathDomain mathDomain>
	class QuadraticCostFunction final: public CostFunction<mathDomain>
	{
	public:
		using CostFunction<mathDomain>::CostFunction;
		
		constexpr CostFunctionType GetType() const noexcept override { return CostFunctionType::Quadratic; }
		
		double EvaluateWorker(typename ICostFunction<mathDomain>::Matrix& expected, const typename ICostFunction<mathDomain>::Matrix& actual) const noexcept override
		{
			expected -= actual;
			double norm2 = expected.EuclideanNorm();
			norm2 *= 0.5 * norm2;

			return norm2;
		}
		
		void EvaluateGradient(typename ICostFunction<mathDomain>::Matrix& expected,
							    const typename ICostFunction<mathDomain>::Matrix& actual,
							    const typename ICostFunction<mathDomain>::Matrix& activationDerivative) const noexcept override
		{
			expected -= actual;
			expected %= activationDerivative;
		}
	};
}
