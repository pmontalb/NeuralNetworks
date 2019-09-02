#pragma once

#include <NeuralNetworks/MiniBatchData.h>
#include <NeuralNetworks/Stopwatch.h>
#include <NeuralNetworks/Initializers/IBiasWeightInitializer.h>
#include <NeuralNetworks/CostFunctions/ICostFunction.h>
#include <NeuralNetworks/Shufflers/IShuffler.h>
#include <NeuralNetworks/Layers/ILayer.h>

namespace nn
{
	template<MathDomain mathDomain>
	class Network
	{
		using mat = Matrix<mathDomain>;
		using vec = Vector<mathDomain>;
		using lay = std::unique_ptr<ILayer<mathDomain>>;
	public:
		explicit Network(const std::vector<size_t>& nNeurons) noexcept;
		
		Network(const std::vector<lay>& layers,
				std::unique_ptr<ICostFunction<mathDomain>>&& costFunction,
				std::unique_ptr<IShuffler<mathDomain>>&& miniBatchShuffler) noexcept;
		
		void Evaluate(mat& out, const mat& in, const int debugLevel = 0) const noexcept;
		
		void Train(const NetworkTrainingData<mathDomain>& networkTrainingData) noexcept;
		
		inline auto GetNumberOfLayers() const noexcept { return _layers.size(); }
		
	private:
		void UpdateMiniBatch(MiniBatchData<mathDomain>& data) noexcept;
		
		void AdjointDifferentiation(MiniBatchData<mathDomain>& data) const noexcept;
		
		void SetCostFunctionGradient(const vec& expectedOutput) const noexcept;
		
		void UpdateBiasesAndWeights(MiniBatchData<mathDomain>& data) noexcept;
	private:
		const std::vector<lay>& _layers;
		const std::unique_ptr<ICostFunction<mathDomain>> _costFunction;
		const std::unique_ptr<IShuffler<mathDomain>> _miniBatchShuffler;
		
	};
}

#include <NeuralNetworks/Network.tpp>
