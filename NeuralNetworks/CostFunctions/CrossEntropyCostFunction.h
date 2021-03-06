#pragma once

#include <NeuralNetworks/CostFunctions/SimpleGradientCostFunction.h>
#include <NeuralNetworks/NeuralNetworksManager.h>

namespace nn
{
	template<MathDomain mathDomain>
	class CrossEntropyCostFunction: public SimpleGradientCostFunction<mathDomain>
	{
	public:
		using SimpleGradientCostFunction<mathDomain>::SimpleGradientCostFunction;
		
		constexpr CostFunctionType GetType() const noexcept override { return CostFunctionType::CrossEntropy; }
		
		double EvaluateWorker(typename ICostFunction<mathDomain>::Matrix& modelOutput, const typename ICostFunction<mathDomain>::Matrix& expectedOutput) const noexcept override
		{
			double cost = 0.0;
			nn::detail::CrossEntropyCostFunction(cost, modelOutput.GetBuffer(), expectedOutput.GetBuffer());
			return cost;
		}
	};
}
