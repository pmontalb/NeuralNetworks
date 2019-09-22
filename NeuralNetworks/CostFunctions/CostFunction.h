#pragma once

#include <NeuralNetworks/CostFunctions/ICostFunction.h>

namespace nn
{
	template<MathDomain mathDomain>
	class CostFunction: public ICostFunction<mathDomain>
	{
	public:
		using ICostFunction<mathDomain>::ICostFunction;
		
		double Evaluate(typename ICostFunction<mathDomain>::Matrix& modelOutput,
				const typename ICostFunction<mathDomain>::Matrix& expectedOutput,
				const NetworkTopology<mathDomain>& topology,
				const double lambda) const noexcept override
		{
			double cost = EvaluateWorker(modelOutput, expectedOutput);
			cost /= modelOutput.nCols();
			
			const double weightCost = topology.EvaluateTotalWeightCost();
			return cost + 0.5 * lambda * weightCost;
		}
	
	protected:
		virtual double EvaluateWorker(typename ICostFunction<mathDomain>::Matrix& activations, const typename ICostFunction<mathDomain>::Matrix& expectedOutput) const noexcept = 0;
	};
}
