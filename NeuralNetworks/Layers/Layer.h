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
			  _biasGradient(nOutput,0.0),
			  _weightGradient(nOutput, nInput, 0.0),
			  
			  _zVector(nOutput, 0.0),
			  _activation(nOutput, 0.0),
			  _activationGradient(nOutput, 0.0),
			  _biasGradientCache(nOutput, 0.0),
			
			  _activationFunction(std::move(activationFunction))
		{
			initializer.Set(_bias);
			initializer.Set(_weight);
		}
		
		virtual ~Layer() = default;
		
		inline size_t GetNumberOfInputs() const noexcept override final { return _nInput; }
		inline size_t GetNumberOfOutputs() const noexcept override final { return _nOutput; }
		
		void Update(const double averageLearningRate, const double regularizationFactor) noexcept override
		{
			_bias.AddEqual(_biasGradient, -averageLearningRate);
			
			_weight.Scale(regularizationFactor);
			_weight.AddEqual(_weightGradient, -averageLearningRate);
		}
		
		void Reset() const noexcept override
		{
			_biasGradient.Set(0.0);
			_biasGradientCache.Set(0.0);
			_weightGradient.Set(0.0);
		}
		
		inline typename ILayer<mathDomain>::Vector& GetActivation() noexcept override final { return _activation; }
		inline const typename ILayer<mathDomain>::Vector& GetActivationGradient() const noexcept override final { return _activationGradient; }
		inline const typename ILayer<mathDomain>::Weight& GetWeight() const noexcept override final { return _weight; }
		inline const typename ILayer<mathDomain>::Bias& GetBias() const noexcept override final { return _bias; }
		inline typename ILayer<mathDomain>::Weight& GetWeightGradient() noexcept override final { return _weightGradient; }
		inline typename ILayer<mathDomain>::Bias& GetBiasGradient() noexcept override final { return _biasGradient; }
		
		// TODO: make it private and accessible only to the optimizer
		inline typename ILayer<mathDomain>::Bias& GetBiasGradientCache() noexcept override final { return _biasGradientCache; }
		
	protected:
		const size_t _nInput;
		const size_t _nOutput;
		
		typename ILayer<mathDomain>::Bias _bias;
		typename ILayer<mathDomain>::Weight _weight;
		
		typename ILayer<mathDomain>::Bias _biasGradient;
		typename ILayer<mathDomain>::Weight _weightGradient;
		
		typename ILayer<mathDomain>::Vector _zVector; // stores weight * input + bias
		typename ILayer<mathDomain>::Vector _activation;
		typename ILayer<mathDomain>::Vector _activationGradient;
		
		typename ILayer<mathDomain>::Bias _biasGradientCache;
		
		std::unique_ptr<IActivationFunction<mathDomain>> _activationFunction;
	};
}
