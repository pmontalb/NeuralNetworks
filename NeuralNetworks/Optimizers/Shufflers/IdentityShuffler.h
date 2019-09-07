#pragma once

#include <Optimizers/Shufflers/IShuffler.h>

namespace nn
{
	template<MathDomain mathDomain>
	class IdentityShuffler final: public IShuffler<mathDomain>
	{
	public:
		using IShuffler<mathDomain>::IShuffler;
		void Shuffle(typename IShuffler<mathDomain>::Matrix&, typename IShuffler<mathDomain>::Matrix&) const noexcept override {}  // no shuffle
	};
}
