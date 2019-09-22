#pragma once

#include <NeuralNetworks/CostFunctions/SimpleGradientCostFunction.h>
#include <NeuralNetworksManager.h>

namespace nn
{
	// Cross entropy when using softmax layer!
	template<MathDomain mathDomain>
	class LogLikelihoodCostFunction final: public SimpleGradientCostFunction<mathDomain>
	{
	public:
		using SimpleGradientCostFunction<mathDomain>::SimpleGradientCostFunction;
		
		constexpr CostFunctionType GetType() const noexcept override { return CostFunctionType::LogLikelihood; }
		
		double EvaluateWorker(typename ICostFunction<mathDomain>::Matrix& modelOutput, const typename ICostFunction<mathDomain>::Matrix& expectedOutput) const noexcept override
		{
			double cost = 0.0;
			nn::detail::LogLikelihoodCostFunction(cost, modelOutput.GetBuffer(), expectedOutput.GetBuffer());
			return cost;
		}
	};
}
