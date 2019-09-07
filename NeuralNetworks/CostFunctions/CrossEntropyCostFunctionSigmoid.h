#pragma once

#include <NeuralNetworks/CostFunctions/CostFunction.h>
#include <NeuralNetworksManager.h>

namespace nn
{
	template<MathDomain mathDomain>
	class CrossEntropyCostFunction: public CostFunction<mathDomain>
	{
	public:
		using CostFunction<mathDomain>::CostFunction;
		
		double EvaluateWorker(typename ICostFunction<mathDomain>::Matrix& modelOutput, const typename ICostFunction<mathDomain>::Matrix& expectedOutput) const noexcept override
		{
			double cost = 0.0;
			nn::detail::CrossEntropyCostFunction(cost, modelOutput.GetBuffer(), expectedOutput.GetBuffer());
			return cost;
		}
		
		void EvaluateGradient(typename ICostFunction<mathDomain>::Vector& expected,
								const typename ICostFunction<mathDomain>::Vector& actual,
								const typename ICostFunction<mathDomain>::Vector&) const noexcept override final
		{
			expected -= actual;
		}
	};
}
