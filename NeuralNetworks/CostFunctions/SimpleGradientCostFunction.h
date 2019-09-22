#pragma once

#include <NeuralNetworks/CostFunctions/CostFunction.h>
#include <NeuralNetworksManager.h>

namespace nn
{
	template<MathDomain mathDomain>
	class SimpleGradientCostFunction: public CostFunction<mathDomain>
	{
	public:
		using CostFunction<mathDomain>::CostFunction;
		
		void EvaluateGradient(typename ICostFunction<mathDomain>::Matrix& expected,
		                      const typename ICostFunction<mathDomain>::Matrix& actual,
		                      const typename ICostFunction<mathDomain>::Matrix&) const noexcept override final
		{
			expected -= actual;
		}
	};
}
