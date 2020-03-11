
#include <gtest/gtest.h>
#include <ColumnWiseMatrix.h>
#include <Types.h>

namespace nnt
{
	class DataTests : public ::testing::Test
	{
	};
	
	TEST_F(DataTests, DeserializeSingle)
	{
		const std::string path = getenv("DATA_PATH");
		
		auto trainingInput = cl::ColumnWiseMatrix<MemorySpace::Device, MathDomain::Float>::MatrixFromBinaryFile(path + "/Data/TrainingInputSingle.npy", false, true);
		ASSERT_EQ(trainingInput.nRows(), 784);
		ASSERT_EQ(trainingInput.nCols(), 50000);
		
		auto trainingOutput = cl::ColumnWiseMatrix<MemorySpace::Device, MathDomain::Float>::MatrixFromBinaryFile(path + "/Data/TrainingOutputSingle.npy", false, true);
		ASSERT_EQ(trainingOutput.nRows(), 10);
		ASSERT_EQ(trainingOutput.nCols(), 50000);
		
		auto testInput = cl::ColumnWiseMatrix<MemorySpace::Device, MathDomain::Float>::MatrixFromBinaryFile(path + "/Data/TestInputSingle.npy", false, true);
		ASSERT_EQ(testInput.nRows(), 784);
		ASSERT_EQ(testInput.nCols(), 10000);
		
		auto testOutput = cl::ColumnWiseMatrix<MemorySpace::Device, MathDomain::Float>::MatrixFromBinaryFile(path + "/Data/TestOutputSingle.npy", false, true);
		ASSERT_EQ(testOutput.nRows(), 10);
		ASSERT_EQ(testOutput.nCols(), 10000);
		
		auto validationInput = cl::ColumnWiseMatrix<MemorySpace::Device, MathDomain::Float>::MatrixFromBinaryFile(path + "/Data/ValidationInputSingle.npy", false, true);
		ASSERT_EQ(validationInput.nRows(), 784);
		ASSERT_EQ(validationInput.nCols(), 10000);
		
		auto validationOutput = cl::ColumnWiseMatrix<MemorySpace::Device, MathDomain::Float>::MatrixFromBinaryFile(path + "/Data/ValidationOutputSingle.npy", false, true);
		ASSERT_EQ(validationOutput.nRows(), 10);
		ASSERT_EQ(validationOutput.nCols(), 10000);
	}
	
	TEST_F(DataTests, DeserializeDouble)
	{
		const std::string path = getenv("DATA_PATH");
		
		auto trainingInput = cl::ColumnWiseMatrix<MemorySpace::Device, MathDomain::Double>::MatrixFromBinaryFile(path + "/Data/TrainingInputDouble.npy", false, true);
		ASSERT_EQ(trainingInput.nRows(), 784);
		ASSERT_EQ(trainingInput.nCols(), 50000);
		
		auto trainingOutput = cl::ColumnWiseMatrix<MemorySpace::Device, MathDomain::Double>::MatrixFromBinaryFile(path + "/Data/TrainingOutputDouble.npy", false,true);
		ASSERT_EQ(trainingOutput.nRows(), 10);
		ASSERT_EQ(trainingOutput.nCols(), 50000);
		
		auto testInput = cl::ColumnWiseMatrix<MemorySpace::Device, MathDomain::Double>::MatrixFromBinaryFile(path + "/Data/TestInputDouble.npy", false, true);
		ASSERT_EQ(testInput.nRows(), 784);
		ASSERT_EQ(testInput.nCols(), 10000);
		
		auto testOutput = cl::ColumnWiseMatrix<MemorySpace::Device, MathDomain::Double>::MatrixFromBinaryFile(path + "/Data/TestOutputDouble.npy",false, true);
		ASSERT_EQ(testOutput.nRows(), 10);
		ASSERT_EQ(testOutput.nCols(), 10000);
		
		auto validationInput = cl::ColumnWiseMatrix<MemorySpace::Device, MathDomain::Double>::MatrixFromBinaryFile(path + "/Data/ValidationInputDouble.npy", false,true);
		ASSERT_EQ(validationInput.nRows(), 784);
		ASSERT_EQ(validationInput.nCols(), 10000);
		
		auto validationOutput = cl::ColumnWiseMatrix<MemorySpace::Device, MathDomain::Double>::MatrixFromBinaryFile(path + "/Data/ValidationOutputDouble.npy", false,true);
		ASSERT_EQ(validationOutput.nRows(), 10);
		ASSERT_EQ(validationOutput.nCols(), 10000);
	}
}
