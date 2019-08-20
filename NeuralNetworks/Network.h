#pragma once

#include <NeuralNetworks/MiniBatchData.h>
#include <NeuralNetworks/Stopwatch.h>

namespace nn
{
	template<MathDomain mathDomain>
	class Network
	{
		using mat = Matrix<mathDomain>;
		using vec = Vector<mathDomain>;
	public:
		explicit Network(const std::vector<size_t>& nNeurons) noexcept;
		
		void Evaluate(mat& out, const NetworkTrainingData<mathDomain>& networkTrainingData, std::vector<vec>& cache = {}) const noexcept;
		
		void Train(const NetworkTrainingData<mathDomain>& networkTrainingData) noexcept;
		
		inline auto GetNumberOfLayers() const noexcept { return _nNeurons.size() - 1; }
		
	private:
		void UpdateMiniBatch(MiniBatchData<mathDomain>& data) noexcept;
		
		void AdjointDifferentiation(MiniBatchData<mathDomain>& data) const noexcept;
		
		void SetObjectiveFunctionDerivative(const size_t i, MiniBatchData<mathDomain>& data) const noexcept;
		
		void UpdateBiasesAndWeights(MiniBatchData<mathDomain>& data) noexcept;
	private:
		const std::vector<size_t>& _nNeurons;
		std::vector<vec> _biases {};
		std::vector<mat> _weights {};
		
		std::vector<vec> _cache {};
	};
}

#include <NeuralNetworks/Network.tpp>