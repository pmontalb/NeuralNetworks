#include <NeuralNetworksManager.h>
#include <CudaException.h>

#pragma region Macro helpers

#define __CREATE_FUNCTION_0_ARG(NAME, EXCEPTION)\
	EXTERN_C int _##NAME();\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME()\
			{\
				int err = _##NAME();\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_1_ARG(NAME, EXCEPTION, TYPE0, ARG0)\
	EXTERN_C int _##NAME(TYPE0 ARG0);\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0)\
			{\
				int err = _##NAME(ARG0);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_2_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1);\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1)\
			{\
				int err = _##NAME(ARG0, ARG1);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_3_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2);\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_4_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3);\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_5_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4);\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_6_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5);\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_7_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6);\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_8_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6, TYPE7, ARG7)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7);\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#pragma endregion

__CREATE_FUNCTION_2_ARG(Sigmoid, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x)
__CREATE_FUNCTION_3_ARG(SigmoidPrime, CudaKernelExceptionFactory,MemoryBuffer&, z, const MemoryBuffer&, x, const MemoryBuffer&, sigmoid)

__CREATE_FUNCTION_2_ARG(HyperbolicTangent, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x)
__CREATE_FUNCTION_2_ARG(HyperbolicTangentPrime, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x)

__CREATE_FUNCTION_2_ARG(RectifiedLinearUnit, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x)
__CREATE_FUNCTION_2_ARG(RectifiedLinearUnitPrime, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x)

__CREATE_FUNCTION_2_ARG(LeakyRectifiedLinearUnit, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x)
__CREATE_FUNCTION_2_ARG(LeakyRectifiedLinearUnitPrime, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x)

__CREATE_FUNCTION_2_ARG(InverseSquareRootLinearUnit, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x)
__CREATE_FUNCTION_2_ARG(InverseSquareRootLinearUnitPrime, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x)

__CREATE_FUNCTION_2_ARG(ExponentialLinearUnit, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x)
__CREATE_FUNCTION_2_ARG(ExponentialLinearUnitPrime, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x)

__CREATE_FUNCTION_2_ARG(BentIdentity, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x)
__CREATE_FUNCTION_2_ARG(BentIdentityPrime, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x)

__CREATE_FUNCTION_4_ARG(SoftMax, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x, MemoryBuffer&, columnWiseCache, MemoryBuffer&, onesCache)

__CREATE_FUNCTION_3_ARG(CrossEntropyCostFunction, CudaKernelExceptionFactory, double&, cost, MemoryBuffer&, x, const MemoryBuffer&, y)
__CREATE_FUNCTION_3_ARG(LogLikelihoodCostFunction, CudaKernelExceptionFactory, double&, cost, MemoryBuffer&, z, const MemoryBuffer&, x)

#pragma region Undef macros

#undef __CREATE_FUNCTION_0_ARG
#undef __CREATE_FUNCTION_1_ARG
#undef __CREATE_FUNCTION_2_ARG
#undef __CREATE_FUNCTION_3_ARG
#undef __CREATE_FUNCTION_4_ARG
#undef __CREATE_FUNCTION_5_ARG
#undef __CREATE_FUNCTION_6_ARG
#undef __CREATE_FUNCTION_7_ARG
#undef __CREATE_FUNCTION_8_ARG

#pragma endregion
