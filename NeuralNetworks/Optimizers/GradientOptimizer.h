#pragma once

#include <NeuralNetworks/Optimizers/IOptimizer.h>
#include <VectorCollection.h>
#include <ColumnWiseMatrixCollection.h>

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
			, _biasGradients(topology.GetNumberOfOutputs())
			, _weightGradients(topology.GetTransposedSizes())
		{
//			_biasGradients.reserve(_topology.GetSize());
//			_weightGradients.reserve(_topology.GetSize());
//			for (const auto& layer: _topology)
//			{
////				_biasGradients.emplace_back(Vector<mathDomain>(static_cast<unsigned>(layer->GetNumberOfOutputs()), 0.0));
//				_weightGradients.emplace_back(Matrix<mathDomain>(static_cast<unsigned>(layer->GetNumberOfOutputs()), static_cast<unsigned>(layer->GetNumberOfInputs()), 0.0));
////			}
		}
		
		const ICostFunction<mathDomain>& GetCostFunction() const noexcept override final { return *_costFunction; }
	
	protected:
		const NetworkTopology<mathDomain>& _topology;
		const std::unique_ptr<ICostFunction<mathDomain>> _costFunction;
		
		cl::VectorCollection<MemorySpace::Device, mathDomain> _biasGradients;
		cl::ColumnWiseMatrixCollection<MemorySpace::Device, mathDomain> _weightGradients;
	};
}
