#pragma once

namespace nn
{
	template<MathDomain mathDomain> struct NetworkTrainingData;
	template<MathDomain mathDomain> class ICostFunction;
	
	template<MathDomain mathDomain>
	class IOptimizer
	{
	public:
		using Bias = cl::Vector<MemorySpace::Device, mathDomain>;
		using Weight = cl::Vector<MemorySpace::Device, mathDomain>;
		virtual ~IOptimizer() = default;
		
		virtual void Train(const NetworkTrainingData<mathDomain>& networkTrainingData) noexcept = 0;
		virtual const ICostFunction<mathDomain>& GetCostFunction() const noexcept = 0;
	};
}
