#pragma once

#include <Common.cuh>
#include <Flags.cuh>
#include <Types.h>

EXTERN_C
{
	/**
	* Sigmoid(x) = 1.0 / (1.0 + e^(-x))
	*/
	EXPORT int _Sigmoid(MemoryBuffer& z, const MemoryBuffer& x);
	inline EXPORT int _SigmoidRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		return _Sigmoid(_z, _x);
	}
	
	/**
	* Sigmoid'(x) = Sigmoid(x) * (1.0 - Sigmoid(x))
	*/
	EXPORT int _SigmoidPrime(MemoryBuffer& z, const MemoryBuffer& x);
	inline EXPORT int _SigmoidPrimeRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		return _SigmoidPrime(_z, _x);
	}

	/**
	* tanh(z) = 2 * sigmoid(2 * z) - 1
	*/
	EXPORT int _HyperbolicTangent(MemoryBuffer& z, const MemoryBuffer& x);
	inline EXPORT int __HyperbolicTangentRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		return _HyperbolicTangent(_z, _x);
	}
	
	/**
	* tanh'(z) = 1 - tanh^2(z)
	*/
	EXPORT int _HyperbolicTangentPrime(MemoryBuffer& z, const MemoryBuffer& x);
	inline EXPORT int _HyperbolicTangentPrimeRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		return _HyperbolicTangentPrime(_z, _x);
	}

	/**
	* ReLu(z) = max(0, z)
	*/
	EXPORT int _RectifiedLinearUnit(MemoryBuffer& z, const MemoryBuffer& x);
	inline EXPORT int _RectifiedLinearUnitRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		return _RectifiedLinearUnit(_z, _x);
	}

	/**
	* ReLu'(z) = \xi(z > 0)
	*/
	EXPORT int _RectifiedLinearUnitPrime(MemoryBuffer& z, const MemoryBuffer& x);
	inline EXPORT int _RectifiedLinearUnitPrimeRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		return _RectifiedLinearUnitPrime(_z, _x);
	}

	/**
	* Leaky ReLu(z) = \xi(z >= 0) * z + \xi(z < 0) * 0.01 * z
	*/
	EXPORT int _LeakyRectifiedLinearUnit(MemoryBuffer& z, const MemoryBuffer& x);
	inline EXPORT int _LeakyRectifiedLinearUnitRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		return _LeakyRectifiedLinearUnit(_z, _x);
	}
	
	/**
	* ReLu'(z) = \xi(z > 0)
	*/
	EXPORT int _LeakyRectifiedLinearUnitPrime(MemoryBuffer& z, const MemoryBuffer& x);
	inline EXPORT int _LeakyRectifiedLinearUnitPrimeRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		return _LeakyRectifiedLinearUnitPrime(_z, _x);
	}
	
	/**
	* IsrLu(z) = \xi(z >= 0) * z + \xi(z < 0) * z / (sqrt(1 + z^2))
	*/
	EXPORT int _InverseSquareRootLinearUnit(MemoryBuffer& z, const MemoryBuffer& x);
	inline EXPORT int _InverseSquareRootLinearUnitRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		return _InverseSquareRootLinearUnit(_z, _x);
	}

	/**
	* IsrLu'(z) = \xi(z >= 0) * 1 + \xi(z < 0) * (z / (sqrt(1 + z^2)))^3
	*/
	EXPORT int _InverseSquareRootLinearUnitPrime(MemoryBuffer& z, const MemoryBuffer& x);
	inline EXPORT int _InverseSquareRootLinearUnitPrimeRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		return _InverseSquareRootLinearUnitPrime(_z, _x);
	}

	/**
	* ELU(z) = \xi(z >= 0) * z + \xi(z < 0) * (e^z - 1)
	*/
	EXPORT int _ExponentialLinearUnit(MemoryBuffer& z, const MemoryBuffer& x);
	inline EXPORT int _ExponentialLinearUnitRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		return _ExponentialLinearUnit(_z, _x);
	}
	
	/**
	* ELU'(z) = \xi(z >= 0) * 1 + \xi(z < 0) * (z / (sqrt(1 + z^2)))^3
	*/
	EXPORT int _ExponentialLinearUnitPrime(MemoryBuffer& z, const MemoryBuffer& x);
	inline EXPORT int _ExponentialLinearUnitPrimeRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		return _ExponentialLinearUnitPrime(_z, _x);
	}

	/**
	* BI(z) = (sqrt(x^2 + 1) - 1) / 2
	*/
	EXPORT int _BentIdentity(MemoryBuffer& z, const MemoryBuffer& x);
	inline EXPORT int _BentIdentityRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		return _BentIdentity(_z, _x);
	}

	/**
	* BI(z) = 1 + x / [2.0 * (sqrt(x^2 + 1) - 1]
	*/
	EXPORT int _BentIdentityPrime(MemoryBuffer& z, const MemoryBuffer& x);
	inline EXPORT int _BentIdentityPrimeRaw(const ptr_t z, const ptr_t x, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _z(z, size, memorySpace, mathDomain);
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		return _BentIdentity(_z, _x);
	}

	/**
	* BI(z) = (sqrt(x^2 + 1) - 1) / 2
	*/
	EXPORT int _SoftMax(MemoryTile& z, const MemoryTile& x, MemoryBuffer& columnWiseSumCache, MemoryBuffer& onesCache);
	inline EXPORT int _SoftMaxRaw(const ptr_t z, const ptr_t x, const unsigned nRows, const unsigned nCols, const MemorySpace memorySpace, const MathDomain mathDomain, const ptr_t cache = 0, const ptr_t ones = 0)
	{
		MemoryTile _z(z, nRows, nCols, memorySpace, mathDomain);
		MemoryTile _x(x, nRows, nCols, memorySpace, mathDomain);
		MemoryBuffer columnWiseSumCache(cache, nCols, memorySpace, mathDomain);
		MemoryBuffer onesCache(cache, nCols, memorySpace, mathDomain);
		return _SoftMax(_z, _x, columnWiseSumCache, onesCache);
	}

	/**
	* sum(-x * log(y) - (1 - y) * log(1-x))
	* NB: overrides x
	*/
	EXPORT int _CrossEntropyCostFunctionSigmoid(double& cost, MemoryBuffer& x, const MemoryBuffer& y);
	inline EXPORT int _CrossEntropyCostFunctionSigmoidRaw(double& cost, const ptr_t x, const ptr_t y, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		MemoryBuffer _y(y, size, memorySpace, mathDomain);
		return _CrossEntropyCostFunctionSigmoid(cost, _x, _y);
	}

	/**
	* sum(-x * log(y))
	* NB: overrides x
	*/
	EXPORT int _CrossEntropyCostFunctionSoftMax(double& cost, MemoryBuffer& x, const MemoryBuffer& y);
	inline EXPORT int _CrossEntropyCostFunctionSoftMaxRaw(double& cost, const ptr_t x, const ptr_t y, const unsigned size, const MemorySpace memorySpace, const MathDomain mathDomain)
	{
		MemoryBuffer _x(x, size, memorySpace, mathDomain);
		MemoryBuffer _y(y, size, memorySpace, mathDomain);
		return _CrossEntropyCostFunctionSoftMax(cost, _x, _y);
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