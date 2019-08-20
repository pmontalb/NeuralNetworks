#pragma once

#include <NeuralNetworks/TrainingData.h>
#include <Vector.h>

namespace nn
{
	template<MathDomain mathDomain>
	using Vector = cl::Vector<MemorySpace::Device, mathDomain>;
	
	template<MathDomain mathDomain>
	struct MiniBatchData
	{
		const NetworkTrainingData<mathDomain>& networkTrainingData;
		
		std::vector<Vector<mathDomain>> biasGradient {};
		std::vector<Matrix<mathDomain>> weightGradient {};
		size_t startIndex = 0;
		size_t endIndex = 0;
		
		// stores weights[i] * training[j] + biases[i] for j = start, ..., end - 1
		std::vector<Vector<mathDomain>> zVectors {};
		
		// stores sigma(z[i])
		std::vector<Vector<mathDomain>> activations {};
		
		// stores sigma'(z[i])
		std::vector<Vector<mathDomain>> activationsDerivative {};
		
		// helps when calculating the elementwise product, as we keep cumulating on the bias gradient
		std::vector<Vector<mathDomain>> biasGradientCache {};
		
		// helps when calculating the kronecker product, as there's no "+=" version
		std::vector<Matrix<mathDomain>> weightGradientCache;
	
		MiniBatchData(const NetworkTrainingData<mathDomain>& networkTrainingData_,
				      const std::vector<Vector<mathDomain>>& biases,
				      const std::vector<Matrix<mathDomain>>& weights) noexcept;
	
		void Reset() const noexcept;
		
		/// <summary>
		/// z_i = weights * training_j + bias
		/// </summary>
		/// <param name="i"></param>
		/// <param name="j"></param>
		/// <param name="bias"></param>
		/// <param name="weights"></param>
		void EvaluateWorker(const size_t layer, const Vector<mathDomain>& input,
							const Vector<mathDomain>& bias, const Matrix<mathDomain>& weight) noexcept;
	};
}

#include <NeuralNetworks/MiniBatchData.tpp>