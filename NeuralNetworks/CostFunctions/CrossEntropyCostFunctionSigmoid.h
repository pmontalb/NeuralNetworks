#pragma once

#include <NeuralNetworks/CostFunctions/CrossEntropyCostFunction.h>
#include <NeuralNetworksManager.h>

namespace nn
{
	template<MathDomain mathDomain>
	class CrossEntropyCostFunctionSigmoid: public CrossEntropyCostFunction<mathDomain>
	{
	public:
		using CrossEntropyCostFunction<mathDomain>::CrossEntropyCostFunction;
		
		double EvaluateWorker(typename ICostFunction<mathDomain>::Matrix& modelOutput, const typename ICostFunction<mathDomain>::Matrix& expectedOutput) const noexcept override
		{
			double cost = 0.0;
			nn::detail::CrossEntropyCostFunctionSigmoid(cost, modelOutput.GetBuffer(), expectedOutput.GetBuffer());
			return cost;
		}
	};
}
