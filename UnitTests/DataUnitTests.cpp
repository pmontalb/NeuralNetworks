
#include <gtest/gtest.h>
#include <ColumnWiseMatrix.h>
#include <CudaLightKernels/Types.h>

namespace nnt
{
	class DataTests : public ::testing::Test
	{
	};
	
	TEST_F(DataTests, DeserializeSingle)
	{
		const std::string path = getenv("DATA_PATH");
		
		auto trainingInput = cl::MatrixFromBinaryFile<MemorySpace::Device, MathDomain::Float>(path + "/Data/TrainingInputSingle.npy", true);
		ASSERT_EQ(trainingInput.nRows(), 784);
		ASSERT_EQ(trainingInput.nCols(), 50000);
		
		auto trainingOutput = cl::MatrixFromBinaryFile<MemorySpace::Device, MathDomain::Float>(path + "/Data/TrainingOutputSingle.npy", true);
		ASSERT_EQ(trainingOutput.nRows(), 10);
		ASSERT_EQ(trainingOutput.nCols(), 50000);
		
		auto testInput = cl::MatrixFromBinaryFile<MemorySpace::Device, MathDomain::Float>(path + "/Data/TestInputSingle.npy", true);
		ASSERT_EQ(testInput.nRows(), 784);
		ASSERT_EQ(testInput.nCols(), 10000);
		
		auto testOutput = cl::MatrixFromBinaryFile<MemorySpace::Device, MathDomain::Float>(path + "/Data/TestOutputSingle.npy", true);
		ASSERT_EQ(testOutput.nRows(), 10);
		ASSERT_EQ(testOutput.nCols(), 10000);
		
		auto validationInput = cl::MatrixFromBinaryFile<MemorySpace::Device, MathDomain::Float>(path + "/Data/ValidationInputSingle.npy", true);
		ASSERT_EQ(validationInput.nRows(), 784);
		ASSERT_EQ(validationInput.nCols(), 10000);
		
		auto validationOutput = cl::MatrixFromBinaryFile<MemorySpace::Device, MathDomain::Float>(path + "/Data/ValidationOutputSingle.npy", true);
		ASSERT_EQ(validationOutput.nRows(), 10);
		ASSERT_EQ(validationOutput.nCols(), 10000);
	}
	
	TEST_F(DataTests, DeserializeDouble)
	{
		const std::string path = getenv("DATA_PATH");
		
		auto trainingInput = cl::MatrixFromBinaryFile<MemorySpace::Device, MathDomain::Double>(path + "/Data/TrainingInputDouble.npy", true);
		ASSERT_EQ(trainingInput.nRows(), 784);
		ASSERT_EQ(trainingInput.nCols(), 50000);
		
		auto trainingOutput = cl::MatrixFromBinaryFile<MemorySpace::Device, MathDomain::Double>(path + "/Data/TrainingOutputDouble.npy", true);
		ASSERT_EQ(trainingOutput.nRows(), 10);
		ASSERT_EQ(trainingOutput.nCols(), 50000);
		
		auto testInput = cl::MatrixFromBinaryFile<MemorySpace::Device, MathDomain::Double>(path + "/Data/TestInputDouble.npy", true);
		ASSERT_EQ(testInput.nRows(), 784);
		ASSERT_EQ(testInput.nCols(), 10000);
		
		auto testOutput = cl::MatrixFromBinaryFile<MemorySpace::Device, MathDomain::Double>(path + "/Data/TestOutputDouble.npy", true);
		ASSERT_EQ(testOutput.nRows(), 10);
		ASSERT_EQ(testOutput.nCols(), 10000);
		
		auto validationInput = cl::MatrixFromBinaryFile<MemorySpace::Device, MathDomain::Double>(path + "/Data/ValidationInputDouble.npy", true);
		ASSERT_EQ(validationInput.nRows(), 784);
		ASSERT_EQ(validationInput.nCols(), 10000);
		
		auto validationOutput = cl::MatrixFromBinaryFile<MemorySpace::Device, MathDomain::Double>(path + "/Data/ValidationOutputDouble.npy", true);
		ASSERT_EQ(validationOutput.nRows(), 10);
		ASSERT_EQ(validationOutput.nCols(), 10000);
	}
}
