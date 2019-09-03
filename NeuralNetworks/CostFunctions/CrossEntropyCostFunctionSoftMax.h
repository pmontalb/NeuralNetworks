#pragma once

#include <NeuralNetworks/CostFunctions/CostFunction.h>
#include <NeuralNetworksManager.h>

namespace nn
{
	// Cross entropy when using softmax layer!
	template<MathDomain mathDomain>
	class CrossEntropyCostFunctionSoftMax final: public CrossEntropyCostFunction<mathDomain>
	{
	public:
		using CostFunction<mathDomain>::CostFunction;
		
		double EvaluateWorker(typename ICostFunction<mathDomain>::Matrix& modelOutput, const typename ICostFunction<mathDomain>::Matrix& expectedOutput) const noexcept override
		{
			double cost = 0.0;
			nn::detail::CrossEntropyCostFunctionSoftMax(cost, modelOutput.GetBuffer(), expectedOutput.GetBuffer());
			return cost;
		}
	};
}
