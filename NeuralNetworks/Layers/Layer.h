#pragma once

#include <NeuralNetworks/Layers/Initializers/IBiasWeightInitializer.h>
#include <NeuralNetworks/Layers/ILayer.h>
#include <NeuralNetworks/Activations/IActivationFunction.h>
#include <NeuralNetworks/ISerializable.h>

#include <sys/types.h>
#include <unistd.h>

namespace nn
{
	template<MathDomain mathDomain>
	class Layer: public ILayer<mathDomain>
	{
	public:
		Layer(const unsigned nInput,
			  const unsigned nOutput,
			  std::unique_ptr<IActivationFunction<mathDomain>>&& activationFunction,
			  IBiasWeightInitializer<mathDomain>&& initializer)
			: ILayer<mathDomain>(),
			  
			  _nInput(nInput),
			  _nOutput(nOutput),
			  
			  _bias(nOutput, 0.0),
			  _weight(nOutput, nInput, 0.0),
			
			  _activationFunction(std::move(activationFunction))
		{
			initializer.Set(_bias);
			initializer.Set(_weight);
		}
        Layer(const Layer&) = delete;
        Layer& operator=(const Layer&) = delete;
		
		std::ostream& operator <<(std::ostream& stream) const noexcept override
		{
			const char* dataPath = getenv("DATA_PATH");
			if (!dataPath)
				return stream;
			
			stream << ToString(this->GetType()) << std::endl;
			stream << _nInput << std::endl;
			stream << _nOutput << std::endl;
			stream << ToString(this->_activationFunction->GetType()) << std::endl;
			
			const std::string pidStr = std::to_string(getpid());
			
			static size_t id = 0;  // terrible hack...
			
			std::string weightFileName = dataPath;
			weightFileName += "/weight.";
			weightFileName += std::to_string(id) + ".";
			weightFileName += pidStr;
			_weight.ToBinaryFile(weightFileName, true);
			stream << weightFileName << std::endl;
			
			std::string biasFileName = dataPath;
			biasFileName += "/bias.";
			biasFileName += std::to_string(id) + ".";
			biasFileName += pidStr;
			_bias.ToBinaryFile(biasFileName, true);
			stream << biasFileName << std::endl;
			
			++id;
			
			return stream;
		}
		
		std::istream& operator >>(std::istream& stream) noexcept override
		{
			std::string weightFileName;
			std::getline(stream, weightFileName);
			_weight.ReadFrom(cl::ColumnWiseMatrix<MemorySpace::Device, mathDomain>::MatrixFromBinaryFile(weightFileName, true));
			
			std::string biasFileName;
			std::getline(stream, biasFileName);
			_bias.ReadFrom(cl::Vector<MemorySpace::Device, mathDomain>::VectorFromBinaryFile(biasFileName, true));
			
			return stream;
		}
		
		inline size_t GetNumberOfInputs() const noexcept override final { return _nInput; }
		inline size_t GetNumberOfOutputs() const noexcept override final { return _nOutput; }
		
		void Update(const typename ILayer<mathDomain>::Bias& biasGradient,
		            const typename ILayer<mathDomain>::Weight& weightGradient,
		            const double averageLearningRate,
		            const double regularizationFactor) noexcept override final
		{
			_bias.AddEqual(biasGradient, -averageLearningRate);
			_weight.AddEqualMatrix(weightGradient, MatrixOperation::None, MatrixOperation::None, regularizationFactor, -averageLearningRate);
		}
		
		CostFunctionType GetBestCostFunctionType() const noexcept override { return _activationFunction->GetBestCostFunction(); }
		std::unique_ptr<ICostFunction<mathDomain>> GetBestCostFunction() const noexcept override { return nullptr; }
		
		inline typename ILayer<mathDomain>::Matrix& GetActivation() noexcept override final { return *_lastActivation; }
		inline const typename ILayer<mathDomain>::Matrix& GetActivationGradient() const noexcept override final { return *_lastActivationGradient; }
		inline const typename ILayer<mathDomain>::Weight& GetWeight() const noexcept override final { return _weight; }
		inline const typename ILayer<mathDomain>::Bias& GetBias() const noexcept override final { return _bias; }
		
	protected:
		const size_t _nInput;
		const size_t _nOutput;
		
		typename ILayer<mathDomain>::Bias _bias;
		typename ILayer<mathDomain>::Weight _weight;
		
		std::unordered_map<size_t, typename ILayer<mathDomain>::Matrix> _zMatrix {}; // stores weight * input + bias
		std::unordered_map<size_t, typename ILayer<mathDomain>::Matrix> _batchedActivation {};
		std::unordered_map<size_t, typename ILayer<mathDomain>::Matrix> _batchedActivationGradient {}; // TODO: move it into the optimizers?
		
		typename ILayer<mathDomain>::Matrix* _lastActivation = nullptr;
		typename ILayer<mathDomain>::Matrix* _lastActivationGradient = nullptr;
		
		std::unique_ptr<IActivationFunction<mathDomain>> _activationFunction;
	};
}
