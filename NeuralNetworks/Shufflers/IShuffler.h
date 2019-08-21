#pragma once

#include <Types.h>

namespace nn
{
	template<MathDomain mathDomain>
	class IShuffler
	{
	public:
		using Matrix = cl::ColumnWiseMatrix<MemorySpace::Device, mathDomain>;
		
		virtual ~IShuffler() = default;
		virtual void Shuffle(Matrix& input, Matrix& expectedOutput) const noexcept = 0;
	};
}
