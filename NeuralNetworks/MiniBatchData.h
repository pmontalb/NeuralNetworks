#pragma once

#include <NeuralNetworks/TrainingData.h>
#include <Vector.h>

namespace nn
{
	template<MathDomain mathDomain>
	using Vector = cl::Vector<MemorySpace::Device, mathDomain>;
	
	template<MathDomain mathDomain>
	struct MiniBatchData
	{
		const NetworkTrainingData<mathDomain>& networkTrainingData;
		size_t startIndex = 0;
		size_t endIndex = 0;
		
		MiniBatchData(const NetworkTrainingData<mathDomain>& networkTrainingData_) noexcept
			: networkTrainingData(networkTrainingData_)
		{
		}
	};
}
