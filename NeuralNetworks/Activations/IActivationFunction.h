#pragma once

#include <Types.h>
#include <NeuralNetworks/Activations/ActivationFunctionType.h>

namespace nn
{
	template<MathDomain mathDomain>
	class IActivationFunction
	{
	public:
		using Vector = cl::Vector<MemorySpace::Device, mathDomain>;
		virtual ActivationFunctionType GetType() const noexcept = 0;
		
		virtual ~IActivationFunction() = default;
		virtual void Evaluate(Vector& output, const Vector& input) const noexcept = 0;
		virtual void EvaluateGradient(Vector& output, const Vector& input) const noexcept = 0;
	};
}
