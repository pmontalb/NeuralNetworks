#pragma once

#include <Types.h>
#include <NeuralNetworks/CostFunctions/CostFunctionType.h>

namespace nn
{
	template<MathDomain mathDomain> class NetworkTopology;
	
	template<MathDomain mathDomain>
	class ICostFunction
	{
	public:
		using Vector = cl::Vector<MemorySpace::Device, mathDomain>;
		using Matrix = cl::ColumnWiseMatrix<MemorySpace::Device, mathDomain>;
		using Matrices = std::vector<Matrix>;
		
		virtual ~ICostFunction() = default;
		virtual CostFunctionType GetType() const noexcept = 0;
		virtual double Evaluate(Matrix& activations, const Matrix& expectedOutput, const NetworkTopology<mathDomain>& layers, const double lambda) const noexcept = 0;
		virtual void EvaluateGradient(Matrix& expected, const Matrix& actual, const Matrix& activationDerivative) const noexcept = 0;
	};
}
