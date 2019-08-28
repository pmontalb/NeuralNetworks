#include <ObjectiveFunctions.cuh>
#include <CubWrappers.cuh>

template <typename T>
DEVICE T __SigmoidWorker__(const T* RESTRICT x, const unsigned i)
{
	return static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-x[i]));
}

template <typename T>
GLOBAL void __Sigmoid__(T* RESTRICT z, const T* RESTRICT x, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		z[i] = __SigmoidWorker__<T>(x, i);
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __SigmoidPrime__<T>(T* RESTRICT z, const T* RESTRICT x, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		const float sigmoidZ = __SigmoidWorker__<T>(x, i);
		z[i] = sigmoidZ * (static_cast<T>(1.0) - sigmoidZ);
	CUDA_FOR_LOOP_EPILOGUE
}

template <typename T>
GLOBAL void __CrossEntropyCostFunction__(T* RESTRICT x, const T* RESTRICT y, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;
	
	CUDA_FOR_LOOP_PROLOGUE
		const T crossEntropy = -y[i] * log(x[i]) - (1.0 - y[i]) * log(1.0 - x[i]);
		if (!isfinite(crossEntropy))
			x[i] = 0.0;
		else
			x[i] = crossEntropy;
	CUDA_FOR_LOOP_EPILOGUE
}

EXTERN_C
{
	EXPORT int _Sigmoid(MemoryBuffer z, const MemoryBuffer x)
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
	
	EXPORT int _SigmoidPrime(MemoryBuffer z, const MemoryBuffer x)
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

	EXPORT int _CrossEntropyCostFunction(double& cost, MemoryBuffer x, const MemoryBuffer y)
	{
		switch (x.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__CrossEntropyCostFunction__<float>, (float*)x.pointer, (float*)y.pointer, x.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__CrossEntropyCostFunction__<double>, (double*)x.pointer, (double*)y.pointer, x.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		
		// now sum everything together
		return _Sum(cost, x);
	}
}