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
		
		void EvaluateGradient(typename ICostFunction<mathDomain>::Vector& expected,
		                      const typename ICostFunction<mathDomain>::Vector& actual,
		                      const typename ICostFunction<mathDomain>::Vector&) const noexcept override final
		{
			expected -= actual;
		}
	};
}
