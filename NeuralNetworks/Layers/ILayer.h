#pragma once

#include <Types.h>

namespace nn
{
	template<MathDomain mathDomain>
	class ILayer
	{
	public:
		using Weight = cl::ColumnWiseMatrix<MemorySpace::Device, mathDomain>;
		using Bias = cl::Vector<MemorySpace::Device, mathDomain>;
		using Vector = Bias;
		
		virtual ~ILayer() = default;
		
		virtual void Evaluate(const Vector& input, Vector* const output = nullptr) noexcept = 0;
		virtual void Update(const double averageLearningRate, const double regularizationFactor = 0.0) noexcept = 0;
		
		virtual size_t GetNumberOfInputs() const noexcept = 0;
		virtual size_t GetNumberOfOutputs() const noexcept = 0;
		virtual Vector& GetActivation() noexcept = 0;
		virtual const Vector& GetActivationGradient() const noexcept = 0;
		virtual const Weight& GetWeight() const noexcept = 0;
		virtual const Bias& GetBias() const noexcept = 0;
		virtual Weight& GetWeightGradient() noexcept = 0;
		virtual Bias& GetBiasGradient() noexcept = 0;
		virtual Bias& GetBiasGradientCache() noexcept = 0;
		
		// reset cached quantities
		virtual void Reset() const noexcept {};
	};
}