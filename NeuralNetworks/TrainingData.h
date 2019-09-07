#pragma once

#include <ColumnWiseMatrix.h>

#include <functional>

namespace nn
{
	template <MathDomain mathDomain> using Matrix = cl::ColumnWiseMatrix<MemorySpace::Device, mathDomain>;
	template <MathDomain mathDomain> using Vector = cl::Vector<MemorySpace::Device, mathDomain>;
	
	template <MathDomain mathDomain>
	struct TrainingData
	{
		Matrix<mathDomain> input;
		Matrix<mathDomain> expectedOutput;
		
		size_t GetLength() const noexcept
		{
			return input.nRows();
		}
		
		size_t GetNumberOfSamples() const noexcept
		{
			return input.nCols();
		}
		
		TrainingData(Matrix<mathDomain>&& input_, Matrix<mathDomain>&& expectedOutput_)
			: input(std::move(input_)), expectedOutput(std::move(expectedOutput_))
		{
		}
	};
	
	struct HyperParameters
	{
		size_t nEpochs = 10;
		size_t miniBacthSize = 100;
		
		double learningRate = 0.1;
		double lambda = 5.0;
		
		double GetAverageLearningRate() const noexcept
		{
			return learningRate / miniBacthSize;
		}
	};
	
	
	template <MathDomain mathDomain>
	struct NetworkTrainingData
	{
		TrainingData<mathDomain>& trainingData;
		TrainingData<mathDomain>& testData;
		TrainingData<mathDomain>& validationData;
		
		const std::function<double(Matrix<mathDomain>&, const Matrix<mathDomain>&)>& evaluator;
		
		HyperParameters hyperParameters {};
		
		// define every how many epochs to evaluate the network. If it's 0, it evaluates only at the end
		size_t epochCalculationAccuracyTestData = 0;
		size_t epochCalculationAccuracyValidationData = 0;
		size_t epochCalculationAccuracyTrainingData = 0;
		
		size_t epochCalculationTotalCostTestData = 0;
		size_t epochCalculationTotalCostValidationData = 0;
		size_t epochCalculationTotalCostTrainingData = 0;
		
		size_t nMaxEpochsWithNoScoreImprovements = 0;
		
		int debugLevel = 0;
		
		NetworkTrainingData(TrainingData<mathDomain>& trainingData_, TrainingData<mathDomain>& testData_,
		                    TrainingData<mathDomain>& validationData_,
		                    const std::function<double(Matrix<mathDomain>&, const Matrix<mathDomain>&)>& evaluator_,
		                    HyperParameters hyperParameters_ = HyperParameters(),
		                    size_t epochCalculationTestData_ = 0,
		                    size_t epochCalculationValidationData_ = 0,
		                    size_t epochCalculationTrainingData_ = 0,
		                    int debugLevel_ = 0) : trainingData(trainingData_), testData(testData_),
		                                           validationData(validationData_), evaluator(evaluator_),
		                                           hyperParameters(hyperParameters_),
		                                           epochCalculationAccuracyTestData(epochCalculationTestData_),
		                                           epochCalculationAccuracyValidationData(epochCalculationValidationData_),
		                                           epochCalculationAccuracyTrainingData(epochCalculationTrainingData_),
		                                           debugLevel(debugLevel_)
		{
		}
	};
};
