#pragma once

#include <NeuralNetworks/Optimizers/IOptimizer.h>

#include <vector>
#include <memory>

namespace nn
{
	template<MathDomain mathDomain> class NetworkTopology;
	
	template<MathDomain mathDomain>
	class GradientOptimizer: public IOptimizer<mathDomain>
	{
	public:
		GradientOptimizer(const NetworkTopology<mathDomain>& topology, std::unique_ptr<ICostFunction < mathDomain>>&& costFunction) noexcept
			: _topology(topology), _costFunction(std::move(costFunction))
		{
			for (const auto& layer: _topology)
			{
				_biasGradients.emplace_back(Vector<mathDomain>(static_cast<unsigned>(layer->GetNumberOfOutputs()), 0.0));
				_weightGradients.emplace_back(Matrix<mathDomain>(static_cast<unsigned>(layer->GetNumberOfOutputs()), static_cast<unsigned>(layer->GetNumberOfInputs()), 0.0));
			}
		}
		
		const ICostFunction<mathDomain>& GetCostFunction() const noexcept override final { return *_costFunction; }
	
	protected:
		const NetworkTopology<mathDomain>& _topology;
		const std::unique_ptr<ICostFunction<mathDomain>> _costFunction;
		
		std::vector<Vector<mathDomain>> _biasGradients;
		std::vector<Matrix<mathDomain>> _weightGradients;
	};
}
