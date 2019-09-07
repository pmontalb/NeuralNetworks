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
	inline EXPORT int _SigmoidRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _Sigmoid(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}
	
	/**
	* Sigmoid'(z) = Sigmoid(z) * (1.0 - Sigmoid(z))
	*/
	EXPORT int _SigmoidPrime(MemoryBuffer z, MemoryBuffer x);
	inline EXPORT int _SigmoidPrimeRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _SigmoidPrime(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	/**
	* tanh(z) = 2 * sigmoid(2 * z) - 1
	*/
	EXPORT int _HyperbolicTangent(MemoryBuffer z, MemoryBuffer x);
	inline EXPORT int __HyperbolicTangentRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _HyperbolicTangent(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}
	
	/**
	* tanh'(z) = 1 - tanh^2(z)
	*/
	EXPORT int _HyperbolicTangentPrime(MemoryBuffer z, MemoryBuffer x);
	inline EXPORT int _HyperbolicTangentPrimeRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _HyperbolicTangentPrime(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	/**
	* ReLu(z) = max(0, z)
	*/
	EXPORT int _RectifiedLinearUnit(MemoryBuffer z, MemoryBuffer x);
	inline EXPORT int _RectifiedLinearUnitRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _RectifiedLinearUnit(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	/**
	* ReLu'(z) = \xi(z > 0)
	*/
	EXPORT int _RectifiedLinearUnitPrime(MemoryBuffer z, MemoryBuffer x);
	inline EXPORT int _RectifiedLinearUnitPrimeRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _RectifiedLinearUnitPrime(MemoryBuffer(z, size, memorySpace, mathDomain),
		                                 MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	/**
	* Leaky ReLu(z) = \xi(z >= 0) * z + \xi(z < 0) * 0.01 * z
	*/
	EXPORT int _LeakyRectifiedLinearUnit(MemoryBuffer z, MemoryBuffer x);
	inline EXPORT int _LeakyRectifiedLinearUnitRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _LeakyRectifiedLinearUnit(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}
	
	/**
	* ReLu'(z) = \xi(z > 0)
	*/
	EXPORT int _LeakyRectifiedLinearUnitPrime(MemoryBuffer z, MemoryBuffer x);
	inline EXPORT int _LeakyRectifiedLinearUnitPrimeRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _LeakyRectifiedLinearUnitPrime(MemoryBuffer(z, size, memorySpace, mathDomain),
		                                 MemoryBuffer(x, size, memorySpace, mathDomain));
	}
	
	/**
	* IsrLu(z) = \xi(z >= 0) * z + \xi(z < 0) * z / (sqrt(1 + z^2))
	*/
	EXPORT int _InverseSquareRootLinearUnit(MemoryBuffer z, MemoryBuffer x);
	inline EXPORT int _InverseSquareRootLinearUnitRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _InverseSquareRootLinearUnit(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	/**
	* IsrLu'(z) = \xi(z >= 0) * 1 + \xi(z < 0) * (z / (sqrt(1 + z^2)))^3
	*/
	EXPORT int _InverseSquareRootLinearUnitPrime(MemoryBuffer z, MemoryBuffer x);
	inline EXPORT int _InverseSquareRootLinearUnitPrimeRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _InverseSquareRootLinearUnitPrime(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	/**
	* ELU(z) = \xi(z >= 0) * z + \xi(z < 0) * (e^z - 1)
	*/
	EXPORT int _ExponentialLinearUnit(MemoryBuffer z, MemoryBuffer x);
	inline EXPORT int _ExponentialLinearUnitRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _ExponentialLinearUnit(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}
	
	/**
	* ELU'(z) = \xi(z >= 0) * 1 + \xi(z < 0) * (z / (sqrt(1 + z^2)))^3
	*/
	EXPORT int _ExponentialLinearUnitPrime(MemoryBuffer z, MemoryBuffer x);
	inline EXPORT int _ExponentialLinearUnitPrimeRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _ExponentialLinearUnitPrime(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	/**
	* BI(z) = (sqrt(x^2 + 1) - 1) / 2
	*/
	EXPORT int _BentIdentity(MemoryBuffer z, MemoryBuffer x);
	inline EXPORT int _BentIdentityRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _BentIdentity(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	/**
	* BI(z) = 1 + x / [2.0 * (sqrt(x^2 + 1) - 1]
	*/
	EXPORT int _BentIdentityPrime(MemoryBuffer z, MemoryBuffer x);
	inline EXPORT int _BentIdentityPrimeRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _BentIdentity(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	/**
	* BI(z) = (sqrt(x^2 + 1) - 1) / 2
	*/
	EXPORT int _SoftMax(MemoryBuffer z, MemoryBuffer x);
	inline EXPORT int _SoftMaxRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _SoftMax(MemoryBuffer(z, size, memorySpace, mathDomain), MemoryBuffer(x, size, memorySpace, mathDomain));
	}

	/**
	* sum(-x * log(y) - (1 - y) * log(1-x))
	* NB: overrides x
	*/
	EXPORT int _CrossEntropyCostFunctionSigmoid(double& cost, MemoryBuffer x, MemoryBuffer y);
	inline EXPORT int _CrossEntropyCostFunctionSigmoidRaw(double& cost, const ptr_t x, const ptr_t y, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _CrossEntropyCostFunctionSigmoid(cost, MemoryBuffer(x, size, memorySpace, mathDomain), MemoryBuffer(y, size, memorySpace, mathDomain));
	}

	/**
	* sum(-x * log(y))
	* NB: overrides x
	*/
	EXPORT int _CrossEntropyCostFunctionSoftMax(double& cost, MemoryBuffer x, MemoryBuffer y);
	inline EXPORT int _CrossEntropyCostFunctionSoftMaxRaw(double& cost, const ptr_t x, const ptr_t y, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		return _CrossEntropyCostFunctionSoftMax(cost, MemoryBuffer(x, size, memorySpace, mathDomain), MemoryBuffer(y, size, memorySpace, mathDomain));
	}
}

template <typename T>
DEVICE T __SigmoidWorker__(const T x);

template <typename T>
DEVICE T __HyperbolicTangentWorker__(const T x);

template <typename T>
DEVICE T __InverseSquareRootLinearUnitDenominatorWorker__(const T x);

template <typename T>
DEVICE T __ExponentialLinearUnitPrimeWorker__(const T x);

template <typename T>
DEVICE T __BentIdentityPrimeWorker__(const T x);

template <typename T>
DEVICE T __CrossEntropyWorker__(const T x, const T y);

template <typename T>
GLOBAL void __Sigmoid__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz);

template <typename T>
GLOBAL void __SigmoidPrime__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz);

template <typename T>
GLOBAL void __HyperbolicTangent__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz);

template <typename T>
GLOBAL void __HyperbolicTangentPrime__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz);

template <typename T>
GLOBAL void __RectifiedLinearUnit__(T* RESTRICT z, const T* RESTRICT x, const T alpha, const unsigned sz);

template <typename T>
GLOBAL void __RectifiedLinearUnitPrime__(T* RESTRICT z, const T* RESTRICT x, const T alpha, const unsigned sz);

template <typename T>
GLOBAL void __InverseSquareRootLinearUnit__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz);

template <typename T>
GLOBAL void __InverseSquareRootLinearUnitPrime__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz);

template <typename T>
GLOBAL void __ExponentialLinearUnit__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz);

template <typename T>
GLOBAL void __ExponentialLinearUnitPrime__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz);

template <typename T>
GLOBAL void __BentIdentity__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz);

template <typename T>
GLOBAL void __BentIdentityPrime__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz);

template <typename T>
GLOBAL void __SoftMax__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz);

template <typename T>
GLOBAL void __CrossEntropyCostFunctionSigmoid__(T* RESTRICT x, const T* RESTRICT y, const unsigned sz);

template <typename T>
GLOBAL void __CrossEntropyCostFunctionSoftMax__(T* RESTRICT x, const T* RESTRICT y, const unsigned sz);