#include <ObjectiveFunctions.cuh>
#include <CubWrappers.cuh>
#include <CuBlasWrappers.cuh>
#include <MemoryManager.cuh>
#include <BufferInitializer.cuh>

template <typename T>
DEVICE T __SigmoidWorker__(const T x)
{
	return static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-x));
}

template <typename T>
DEVICE T __HyperbolicTangentWorker__(const T x)
{
	return static_cast<T>(2.0) * __SigmoidWorker__<T>(static_cast<T>(2.0) * x) - static_cast<T>(1.0);
}

template <typename T>
DEVICE T __InverseSquareRootLinearUnitDenominatorWorker__(const T x)
{
	return static_cast<T>(1.0) / (static_cast<T>(1.0) + x * x);
}

template <typename T>
DEVICE T __ExponentialLinearUnitPrimeWorker__(const T x)
{
	return exp(x);
}

template <typename T>
DEVICE T __BentIdentityPrimeWorker__(const T x)
{
	return sqrt(x * x + static_cast<T>(1.0));
}

template <typename T>
DEVICE T __CrossEntropyWorker__(const T x, const T y)
{
	return 	x * log(y);
}

template <typename T>
GLOBAL void __Sigmoid__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		z[i] = __SigmoidWorker__<T>(x[i]);
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __SigmoidPrime__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		const T sigmoidZ = __SigmoidWorker__<T>(x[i]);
		z[i] = sigmoidZ * (static_cast<T>(1.0) - sigmoidZ);
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __HyperbolicTangent__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		z[i] = __HyperbolicTangentWorker__<T>(x[i]);
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __HyperbolicTangentPrime__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		const T tanhZ = __HyperbolicTangentWorker__<T>(x[i]);
		z[i] = static_cast<T>(1.0) - tanhZ * tanhZ;
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __RectifiedLinearUnit__(T* RESTRICT z, const T* RESTRICT x, const T alpha, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		if (x[i] <= static_cast<T>(0.0))
			z[i] = alpha * x[i];
		else
			z[i] = x[i];
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __RectifiedLinearUnitPrime__(T* RESTRICT z, const T* RESTRICT x, const T alpha, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		if (x[i] <= static_cast<T>(0.0))
			z[i] = alpha;
		else
			z[i] = 1.0;
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __InverseSquareRootLinearUnit__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		if (x[i] <= static_cast<T>(0.0))
			z[i] = x[i] * __InverseSquareRootLinearUnitDenominatorWorker__(x[i]);
		else
			z[i] = x[i];
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __InverseSquareRootLinearUnitPrime__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		if (x[i] <= static_cast<T>(0.0))
		{
			const double factor = __InverseSquareRootLinearUnitDenominatorWorker__(x[i]);
			z[i] = factor * factor * factor;
		}
		else
			z[i] = 1.0;
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __ExponentialLinearUnit__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		if (x[i] <= static_cast<T>(0.0))
			z[i] = __ExponentialLinearUnitPrimeWorker__(x[i]) - static_cast<T>(1.0);
		else
			z[i] = x[i];
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __ExponentialLinearUnitPrime__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		if (x[i] <= static_cast<T>(0.0))
			z[i] = __ExponentialLinearUnitPrimeWorker__(x[i]);
		else
			z[i] = x[i];
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __BentIdentity__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		const T sqrtX2 = __BentIdentityPrimeWorker__(x[i]);
		z[i] = x[i] + static_cast<T>(0.5) * (sqrtX2 - static_cast<T>(1.0));
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __BentIdentityPrime__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		const T sqrtX2 = __BentIdentityPrimeWorker__(x[i]);
		z[i] = static_cast<T>(1.0) + static_cast<T>(0.5) * x[i] / sqrtX2;
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __SoftMax__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		z[i] = exp(x[i]);  // normalised later on!
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __CrossEntropyCostFunctionSigmoid__(T* RESTRICT x, const T* RESTRICT y, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		const T crossEntropy = -__CrossEntropyWorker__(y[i], x[i]) - __CrossEntropyWorker__(1.0 - y[i], 1.0 - x[i]);
		if (!isfinite(crossEntropy))
			x[i] = 0.0;
		else
			x[i] = crossEntropy;
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __CrossEntropyCostFunctionSoftMax__(T* RESTRICT x, const T* RESTRICT y, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		const T crossEntropy = -__CrossEntropyWorker__(y[i], x[i]);
		if (!isfinite(crossEntropy))
			x[i] = 0.0;
		else
			x[i] = crossEntropy;
	CUDA_FOR_LOOP_EPILOGUE
}


static inline int RectifiedLinearUnitWorker(MemoryBuffer& z, const MemoryBuffer& x, const double alpha)
{
	switch (z.mathDomain)
	{
		case MathDomain::Float:
			CUDA_CALL_SINGLE(__RectifiedLinearUnit__<float>, (float*)z.pointer, (float*)x.pointer, (float)alpha, z.size);
			break;
		case MathDomain::Double:
			CUDA_CALL_DOUBLE(__RectifiedLinearUnit__<double>, (double*)z.pointer, (double*)x.pointer, (double)alpha, z.size);
			break;
		default:
			return CudaKernelException::_NotImplementedException;
	}
	return cudaGetLastError();
}

static inline int RectifiedLinearUnitPrimeWorker(MemoryBuffer& z, const MemoryBuffer& x, const double alpha)
{
	switch (z.mathDomain)
	{
		case MathDomain::Float:
			CUDA_CALL_SINGLE(__RectifiedLinearUnitPrime__<float>, (float*)z.pointer, (float*)x.pointer, (float)alpha, z.size);
			break;
		case MathDomain::Double:
			CUDA_CALL_DOUBLE(__RectifiedLinearUnitPrime__<double>, (double*)z.pointer, (double*)x.pointer, (double)alpha, z.size);
			break;
		default:
			return CudaKernelException::_NotImplementedException;
	}
	return cudaGetLastError();
}

EXTERN_C
{
	EXPORT int _Sigmoid(MemoryBuffer& z, const MemoryBuffer& x)
	{
		switch (z.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__Sigmoid__<float>, (float*)z.pointer, (float*)x.pointer, z.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__Sigmoid__<double>, (double*)z.pointer, (double*)x.pointer, z.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
	}
	
	EXPORT int _SigmoidPrime(MemoryBuffer& z, const MemoryBuffer& x)
	{
		switch (z.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__SigmoidPrime__<float>, (float*) z.pointer, (float*) x.pointer, z.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__SigmoidPrime__<double>, (double*) z.pointer, (double*) x.pointer, z.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
	}

	EXPORT int _HyperbolicTangent(MemoryBuffer& z, const MemoryBuffer& x)
	{
		switch (z.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__HyperbolicTangent__<float>, (float*)z.pointer, (float*)x.pointer, z.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__HyperbolicTangent__<double>, (double*)z.pointer, (double*)x.pointer, z.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
	}
	
	EXPORT int _HyperbolicTangentPrime(MemoryBuffer& z, const MemoryBuffer& x)
	{
		switch (z.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__HyperbolicTangentPrime__<float>, (float*) z.pointer, (float*) x.pointer, z.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__HyperbolicTangentPrime__<double>, (double*) z.pointer, (double*) x.pointer, z.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
	}

	EXPORT int _RectifiedLinearUnit(MemoryBuffer& z, const MemoryBuffer& x)
	{
		return RectifiedLinearUnitWorker(z, x, 0.0);
	}

	EXPORT int _RectifiedLinearUnitPrime(MemoryBuffer& z, const MemoryBuffer& x)
	{
		return RectifiedLinearUnitPrimeWorker(z, x, 0.0);
	}

	EXPORT int _LeakyRectifiedLinearUnit(MemoryBuffer& z, const MemoryBuffer& x)
	{
		return RectifiedLinearUnitWorker(z, x, 0.01);
	}
	
	EXPORT int _LeakyRectifiedLinearUnitPrime(MemoryBuffer& z, const MemoryBuffer& x)
	{
		return RectifiedLinearUnitPrimeWorker(z, x, 0.01);
	}

	EXPORT int _InverseSquareRootLinearUnit(MemoryBuffer& z, const MemoryBuffer& x)
	{
		switch (z.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__InverseSquareRootLinearUnit__<float>, (float*)z.pointer, (float*)x.pointer, z.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__InverseSquareRootLinearUnit__<double>, (double*)z.pointer, (double*)x.pointer, z.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
	}
	
	EXPORT int _InverseSquareRootLinearUnitPrime(MemoryBuffer& z, const MemoryBuffer& x)
	{
		switch (z.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__InverseSquareRootLinearUnitPrime__<float>, (float*) z.pointer, (float*) x.pointer, z.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__InverseSquareRootLinearUnitPrime__<double>, (double*) z.pointer, (double*) x.pointer, z.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
	}

	EXPORT int _ExponentialLinearUnit(MemoryBuffer& z, const MemoryBuffer& x)
	{
		switch (z.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__InverseSquareRootLinearUnit__<float>, (float*)z.pointer, (float*)x.pointer, z.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__InverseSquareRootLinearUnit__<double>, (double*)z.pointer, (double*)x.pointer, z.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
	}
	
	EXPORT int _ExponentialLinearUnitPrime(MemoryBuffer& z, const MemoryBuffer& x)
	{
		switch (z.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__InverseSquareRootLinearUnitPrime__<float>, (float*) z.pointer, (float*) x.pointer, z.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__InverseSquareRootLinearUnitPrime__<double>, (double*) z.pointer, (double*) x.pointer, z.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
	}

	EXPORT int _BentIdentity(MemoryBuffer& z, const MemoryBuffer& x)
	{
		switch (z.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__BentIdentity__<float>, (float*)z.pointer, (float*)x.pointer, z.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__BentIdentity__<double>, (double*)z.pointer, (double*)x.pointer, z.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
	}
	
	EXPORT int _BentIdentityPrime(MemoryBuffer& z, const MemoryBuffer& x)
	{
		switch (z.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__BentIdentityPrime__<float>, (float*) z.pointer, (float*) x.pointer, z.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__BentIdentityPrime__<double>, (double*) z.pointer, (double*) x.pointer, z.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
	}

	EXPORT int _SoftMax(MemoryTile& z, const MemoryTile& x, MemoryBuffer& columnWiseSumCache, MemoryBuffer& onesCache)
	{
		switch (z.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__SoftMax__<float>, (float*)z.pointer, (float*)x.pointer, z.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__SoftMax__<double>, (double*)z.pointer, (double*)x.pointer, z.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		
		if (columnWiseSumCache.size != z.nCols)
		{
			if (columnWiseSumCache.pointer != 0)
				_Free(columnWiseSumCache);
			columnWiseSumCache.pointer = 0;
		}
		
		if (columnWiseSumCache.pointer == 0)
		{
			columnWiseSumCache = MemoryBuffer(0, z.nCols, z.memorySpace, z.mathDomain);
			_Alloc(columnWiseSumCache);
		}
		
		int err = _RowWiseSum(columnWiseSumCache, z, onesCache, MatrixOperation::Transpose);
		if (err)
			return err;
		
		err = _Reciprocal(columnWiseSumCache);
		if (err)
			return err;
		
		err = _ScaleColumns(z, columnWiseSumCache);
		if (err)
			return err;
		
		return cudaGetLastError();
	}

	EXPORT int _CrossEntropyCostFunctionSigmoid(double& cost, MemoryBuffer& x, const MemoryBuffer& y)
	{
		switch (x.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__CrossEntropyCostFunctionSigmoid__<float>, (float*)x.pointer, (float*)y.pointer, x.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__CrossEntropyCostFunctionSigmoid__<double>, (double*)x.pointer, (double*)y.pointer, x.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		
		// now sum everything together
		return _Sum(cost, x);
	}

	EXPORT int _CrossEntropyCostFunctionSoftMax(double& cost, MemoryBuffer& x, const MemoryBuffer& y)
	{
		switch (x.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__CrossEntropyCostFunctionSoftMax__<float>, (float*)x.pointer, (float*)y.pointer, x.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__CrossEntropyCostFunctionSoftMax__<double>, (double*)x.pointer, (double*)y.pointer, x.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		
		// now sum everything together
		return _Sum(cost, x);
	}
}