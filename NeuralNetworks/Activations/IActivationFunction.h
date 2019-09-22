#pragma once

#include <Types.h>
#include <NeuralNetworks/Activations/ActivationFunctionType.h>
#include <NeuralNetworks/CostFunctions/CostFunctionType.h>

namespace nn
{
	template<MathDomain mathDomain>
	class IActivationFunction
	{
	public:
		using Vector = cl::Vector<MemorySpace::Device, mathDomain>;
		using Matrix = cl::ColumnWiseMatrix<MemorySpace::Device, mathDomain>;
		virtual ActivationFunctionType GetType() const noexcept = 0;
		virtual CostFunctionType GetBestCostFunction() const noexcept = 0;
		
		virtual ~IActivationFunction() = default;
		virtual void Evaluate(Matrix& output, const Matrix& input) const noexcept = 0;
		virtual void EvaluateGradient(Matrix& output, const Matrix& input, const Matrix& activation) const noexcept = 0;
	};
}
