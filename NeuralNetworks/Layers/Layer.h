#pragma once

#include <NeuralNetworks/Initializers/IBiasWeightInitializer.h>
#include <NeuralNetworks/Layers/ILayer.h>
#include <NeuralNetworks/Activations/IActivationFunction.h>

namespace nn
{
	template<MathDomain mathDomain>
	class Layer: public ILayer<mathDomain>
	{
	public:
		Layer(const unsigned nInput,
			  const unsigned nOutput,
			  std::unique_ptr<IActivationFunction<mathDomain>>&& activationFunction,
			  IBiasWeightInitializer<mathDomain>&& initializer)
			: ILayer<mathDomain>(),
			  
			  _nInput(nInput),
			  _nOutput(nOutput),
			  
			  _bias(nOutput, 0.0),
			  _weight(nOutput, nInput, 0.0),
			  
			  _zVector(nOutput, 0.0),
			  _activation(nOutput, 0.0),
			  _activationGradient(nOutput, 0.0),
			
			  _activationFunction(std::move(activationFunction))
		{
			initializer.Set(_bias);
			initializer.Set(_weight);
		}
		
		virtual ~Layer() = default;
		
		inline size_t GetNumberOfInputs() const noexcept override final { return _nInput; }
		inline size_t GetNumberOfOutputs() const noexcept override final { return _nOutput; }
		
		void Update(const typename ILayer<mathDomain>::Bias& biasGradient,
		            const typename ILayer<mathDomain>::Weight& weightGradient,
		            const double averageLearningRate,
		            const double regularizationFactor) noexcept override final
		{
			_bias.AddEqual(biasGradient, -averageLearningRate);
			_weight.AddEqualMatrix(weightGradient, MatrixOperation::None, MatrixOperation::None, regularizationFactor, -averageLearningRate);
		}
		
		std::unique_ptr<ICostFunction<mathDomain>> GetCrossEntropyCostFunction() const noexcept override { return nullptr; }
		
		inline typename ILayer<mathDomain>::Vector& GetActivation() noexcept override final { return _activation; }
		inline const typename ILayer<mathDomain>::Vector& GetActivationGradient() const noexcept override final { return _activationGradient; }
		inline const typename ILayer<mathDomain>::Weight& GetWeight() const noexcept override final { return _weight; }
		inline const typename ILayer<mathDomain>::Bias& GetBias() const noexcept override final { return _bias; }
		
	protected:
		const size_t _nInput;
		const size_t _nOutput;
		
		typename ILayer<mathDomain>::Bias _bias;
		typename ILayer<mathDomain>::Weight _weight;
		
		typename ILayer<mathDomain>::Vector _zVector; // stores weight * input + bias
		typename ILayer<mathDomain>::Vector _activation;
		typename ILayer<mathDomain>::Vector _activationGradient;  // TODO: move it into the optimizers?
		
		std::unique_ptr<IActivationFunction<mathDomain>> _activationFunction;
	};
}
