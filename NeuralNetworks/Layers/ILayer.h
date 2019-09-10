#pragma once

#include <Types.h>
#include <NeuralNetworks/Layers/LayerType.h>
#include <NeuralNetworks/ISerializable.h>

namespace nn
{
	template<MathDomain mathDomain> class IActivationFunction;
	template<MathDomain mathDomain> class ICostFunction;
	
	template<MathDomain mathDomain>
	class ILayer: public ISerializable
	{
	public:
		using Weight = cl::ColumnWiseMatrix<MemorySpace::Device, mathDomain>;
		using Bias = cl::Vector<MemorySpace::Device, mathDomain>;
		using Vector = Bias;
		
		virtual LayerType GetType() const noexcept = 0;
		
		virtual void Evaluate(const Vector& input, Vector* const output = nullptr) noexcept = 0;
		virtual void Update(const typename ILayer<mathDomain>::Bias& biasGradient,
		                    const typename ILayer<mathDomain>::Weight& weightGradient,
		                    const double averageLearningRate,
		                    const double regularizationFactor = 0.0) noexcept = 0;
		virtual std::unique_ptr<ICostFunction<mathDomain>> GetCrossEntropyCostFunction() const noexcept = 0;
		
		virtual size_t GetNumberOfInputs() const noexcept = 0;
		virtual size_t GetNumberOfOutputs() const noexcept = 0;
		virtual Vector& GetActivation() noexcept = 0;
		virtual const Vector& GetActivationGradient() const noexcept = 0;
		virtual const Weight& GetWeight() const noexcept = 0;
		virtual const Bias& GetBias() const noexcept = 0;
		
		// reset cached quantities
		virtual void Reset() const noexcept {}
	};
}
