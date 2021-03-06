#pragma once

#include <Types.h>

#pragma region Macro Utilities

#define __CREATE_FUNCTION_0_ARG(NAME)\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME();\
		}\
	}

#define __CREATE_FUNCTION_1_ARG(NAME, TYPE0, ARG0)\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0);\
		}\
	}

#define __CREATE_FUNCTION_2_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1)\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1);\
		}\
	}

#define __CREATE_FUNCTION_3_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2)\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2);\
		}\
	}

#define __CREATE_FUNCTION_4_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3)\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3);\
		}\
	}

#define __CREATE_FUNCTION_5_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4)\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4);\
		}\
	}

#define __CREATE_FUNCTION_6_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5)\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5);\
		}\
	}

#define __CREATE_FUNCTION_7_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6)\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6);\
		}\
	}

#define __CREATE_FUNCTION_8_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6, TYPE7, ARG7)\
	namespace nn\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7);\
		}\
	}

#pragma endregion

__CREATE_FUNCTION_2_ARG(Sigmoid, MemoryBuffer&, z, const MemoryBuffer&, x)
__CREATE_FUNCTION_3_ARG(SigmoidPrime, MemoryBuffer&, z, const MemoryBuffer&, x, const MemoryBuffer&, sigmoid)

__CREATE_FUNCTION_2_ARG(HyperbolicTangent, MemoryBuffer&, z, const MemoryBuffer&, x)
__CREATE_FUNCTION_2_ARG(HyperbolicTangentPrime, MemoryBuffer&, z, const MemoryBuffer&, x)

__CREATE_FUNCTION_2_ARG(RectifiedLinearUnit, MemoryBuffer&, z, const MemoryBuffer&, x)
__CREATE_FUNCTION_2_ARG(RectifiedLinearUnitPrime, MemoryBuffer&, z, const MemoryBuffer&, x)

__CREATE_FUNCTION_2_ARG(LeakyRectifiedLinearUnit, MemoryBuffer&, z, const MemoryBuffer&, x)
__CREATE_FUNCTION_2_ARG(LeakyRectifiedLinearUnitPrime, MemoryBuffer&, z, const MemoryBuffer&, x)

__CREATE_FUNCTION_2_ARG(InverseSquareRootLinearUnit, MemoryBuffer&, z, const MemoryBuffer&, x)
__CREATE_FUNCTION_2_ARG(InverseSquareRootLinearUnitPrime, MemoryBuffer&, z, const MemoryBuffer&, x)

__CREATE_FUNCTION_2_ARG(ExponentialLinearUnit, MemoryBuffer&, z, const MemoryBuffer&, x)
__CREATE_FUNCTION_2_ARG(ExponentialLinearUnitPrime, MemoryBuffer&, z, const MemoryBuffer&, x)

__CREATE_FUNCTION_2_ARG(BentIdentity, MemoryBuffer&, z, const MemoryBuffer&, x)
__CREATE_FUNCTION_2_ARG(BentIdentityPrime, MemoryBuffer&, z, const MemoryBuffer&, x)

__CREATE_FUNCTION_4_ARG(SoftMax, MemoryBuffer&, z, const MemoryBuffer&, x, MemoryBuffer&, columnWiseCache, MemoryBuffer&, onesCache)

__CREATE_FUNCTION_3_ARG(CrossEntropyCostFunction, double&, cost, MemoryBuffer&, z, const MemoryBuffer&, x)
__CREATE_FUNCTION_3_ARG(LogLikelihoodCostFunction, double&, cost, MemoryBuffer&, z, const MemoryBuffer&, x)

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
