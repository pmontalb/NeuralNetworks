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

	/**
	* sum(-x * log(y) - (1 - y) * log(1-x))
	* NB: overrides x
	*/
	EXPORT int _CrossEntropyCostFunction(double& cost, MemoryBuffer x, MemoryBuffer y);
	EXPORT int _CrossEntropyCostFunctionRaw(double& cost, const ptr_t x, const ptr_t y, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _CrossEntropyCostFunction(cost, MemoryBuffer(x, size, memorySpace, mathDomain), MemoryBuffer(y, size, memorySpace, mathDomain));
	}
}

template <typename T>
DEVICE T __SigmoidWorker__(const T* RESTRICT x, const unsigned i);

template <typename T>
GLOBAL void __Sigmoid__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz);

template <typename T>
GLOBAL void __SigmoidPrime__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz);

template <typename T>
GLOBAL void __CrossEntropyCostFunction__(T* RESTRICT x, const T* RESTRICT y, const unsigned sz);

