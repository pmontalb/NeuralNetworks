#pragma once

#include <cassert>
#include <NeuralNetworks/NeuralNetworksManager.h>

namespace nn
{
	template<MathDomain mathDomain>
	MiniBatchData<mathDomain>::MiniBatchData(const NetworkTrainingData<mathDomain>& networkTrainingData_,
			      const std::vector<Vector<mathDomain>>& biases,
			      const std::vector<Matrix<mathDomain>>& weights) noexcept
		: networkTrainingData(networkTrainingData_)
	{
		for (size_t i = 0; i < biases.size(); ++i)
		{
			biasGradient.emplace_back(Vector<mathDomain>(biases[i].size(), 0.0));
			biasGradientCache.emplace_back(biases[i].size(), 0.0);
			
			weightGradient.emplace_back(Matrix<mathDomain>(weights[i].nRows(), weights[i].nCols(), 0.0));
			weightGradientCache.emplace_back(Matrix<mathDomain>(weights[i].nRows(), weights[i].nCols(), 0.0));
			
			zVectors.emplace_back(Vector<mathDomain>(biases[i].size(), 0.0));
			activations.emplace_back(Vector<mathDomain>(biases[i].size(), 0.0));
			activationsDerivative.emplace_back(Vector<mathDomain>(biases[i].size(), 0.0));
		}
	}
	
	template<MathDomain mathDomain>
	void MiniBatchData<mathDomain>::Reset() const noexcept
	{
		for (size_t i = 0; i < biasGradient.size(); ++i)
			biasGradient[i].Set(0.0);
		
		for (size_t i = 0; i < biasGradientCache.size(); ++i)
			biasGradientCache[i].Set(0.0);
		
		for (size_t i = 0; i < weightGradient.size(); ++i)
			weightGradient[i].Set(0.0);
		
		for (size_t i = 0; i < weightGradientCache.size(); ++i)
			weightGradientCache[i].Set(0.0);
	}
	
	/// <summary>
	/// z_i = weights * training_j + bias
	/// </summary>
	/// <param name="i"></param>
	/// <param name="j"></param>
	/// <param name="bias"></param>
	/// <param name="weights"></param>
	template<MathDomain mathDomain>
	void MiniBatchData<mathDomain>::EvaluateWorker(const size_t layer,
			const Vector<mathDomain>& input,
			const Vector<mathDomain>& bias,
			const Matrix<mathDomain>& weight) noexcept
	{
		assert(zVectors[layer].size() == bias.size());
		
		weight.Dot(zVectors[layer], input);
		zVectors[layer].AddEqual(bias);
		nn::detail::Sigmoid(activations[layer].GetBuffer(), zVectors[layer].GetBuffer());
		nn::detail::SigmoidPrime(activationsDerivative[layer].GetBuffer(), zVectors[layer].GetBuffer());
	}
}
