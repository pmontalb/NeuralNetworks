#pragma once

#include <Optimizers/Shufflers/IShuffler.h>
#include <ColumnWiseMatrix.h>

namespace nn
{
	template<MathDomain mathDomain>
	class RandomShuffler final: public IShuffler<mathDomain>
	{
	public:
		using IShuffler<mathDomain>::IShuffler;
		void Shuffle(typename IShuffler<mathDomain>::Matrix& input, typename IShuffler<mathDomain>::Matrix& expectedOutput) const noexcept override
		{
			IShuffler<mathDomain>::Matrix::RandomShuffleColumnsPair(input, expectedOutput);
		}
	};
}
