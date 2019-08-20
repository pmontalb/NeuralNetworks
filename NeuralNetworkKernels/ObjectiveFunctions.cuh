#pragma once

#include <Common.cuh>
#include <Flags.cuh>
#include <Types.h>

EXTERN_C
{
	/**
	* Sigmoid(z) = 1.0 / (1.0 + e^(-x))
	*/
	EXPORT int _Sigmoid(MemoryBuffer z, MemoryBuffer x);
	EXPORT int _SigmoidRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _Sigmoid(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}
	
	/**
	* Sigmoid'(z) = Sigmoid(z) * (1.0 - Sigmoid(z))
	*/
	EXPORT int _SigmoidPrime(MemoryBuffer z, MemoryBuffer x);
	EXPORT int _SigmoidPrimeRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _SigmoidPrime(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}
}

template <typename T>
DEVICE T __SigmoidWorker__(const T* RESTRICT x, const unsigned i);

template <typename T>
GLOBAL void __Sigmoid__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz);

template <typename T>
GLOBAL void __SigmoidPrime__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz);

