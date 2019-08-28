#pragma once

#include <Types.h>

namespace nn
{
	template<MathDomain mathDomain>
	class ICostFunction
	{
	public:
		using Vector = cl::Vector<MemorySpace::Device, mathDomain>;
		using Matrix = cl::ColumnWiseMatrix<MemorySpace::Device, mathDomain>;
		using Matrices = std::vector<Matrix>;
		
		virtual ~ICostFunction() = default;
		virtual double Evaluate(Matrix& activations, const Matrix& expectedOutput, const Matrices& weights, const double lambda) const noexcept = 0;
		virtual void EvaluateDerivative(Vector& expected, const Vector& actual, const Vector& activationDerivative) const noexcept = 0;
	};
}
