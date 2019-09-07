#pragma once

#include <Optimizers/MiniBatchData.h>
#include <NeuralNetworks/Stopwatch.h>
#include <NeuralNetworks/Initializers/IBiasWeightInitializer.h>
#include <NeuralNetworks/Layers/ILayer.h>

namespace nn
{
	template<MathDomain mathDomain> class IOptimizer;
	
	template<MathDomain mathDomain>
	class Network
	{
		using mat = Matrix<mathDomain>;
		using vec = Vector<mathDomain>;
		using lay = std::unique_ptr<ILayer<mathDomain>>;
	public:
		explicit Network(const std::vector<lay>& layers) noexcept;
		
		void Evaluate(mat& out, const mat& in, const int debugLevel = 0) const noexcept;
		
		void Train(IOptimizer<mathDomain>& optimizer, const NetworkTrainingData<mathDomain>& networkTrainingData) noexcept;
		
		inline decltype(auto) GetLayers() const noexcept { return _layers; }
		inline auto GetNumberOfLayers() const noexcept { return GetLayers().size(); }
		
	private:
		const std::vector<lay>& _layers;
		const std::unique_ptr<IOptimizer<mathDomain>> _optimizer;
		
	};
}

#include <NeuralNetworks/Network.tpp>
