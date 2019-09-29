#pragma once

#include <Types.h>
#include <NeuralNetworks/ISerializable.h>
#include <NeuralNetworks/Layers/LayerFactory.h>
#include <NeuralNetworks/Activations/ActivationFunctionFactory.h>
#include <NeuralNetworks/Layers/Initializers/TrivialBiasWeightInitializer.h>

namespace nn
{
	template <MathDomain mathDomain> class ILayer;
	
	template <MathDomain mathDomain>
	class NetworkTopology final: public ISerializable
	{
		using Layer = std::unique_ptr<ILayer<mathDomain>>;
		using Layers = std::vector<Layer>;
		using Matrix = cl::ColumnWiseMatrix<MemorySpace::Device, mathDomain>;
	public:
		explicit NetworkTopology(Layers&& layers)
		{
			for (auto&& layer: layers)
				_layers.emplace_back(std::move(layer));
			
			for (size_t l = 1; l < _layers.size(); ++l)
				assert(_layers[l]->GetNumberOfInputs() == _layers[l - 1]->GetNumberOfOutputs());
		}
		
		explicit NetworkTopology(std::istream& stream)
		{
			*this >> stream;
		}
		
		virtual ~NetworkTopology() override = default;
		NetworkTopology(const NetworkTopology&) = delete;
		NetworkTopology& operator=(const NetworkTopology&) = delete;
		NetworkTopology(NetworkTopology&&) = default;
		NetworkTopology& operator=(NetworkTopology&&) = default;
		
		inline size_t GetSize() const noexcept { return _layers.size(); }
		inline std::vector<size_t> GetNumberOfOutputs() const noexcept
		{
			std::vector<size_t> ret;
			for (const auto& layer: _layers)
				ret.push_back(layer->GetNumberOfOutputs());
			return ret;
		}
		inline std::vector<std::pair<size_t, size_t>> GetTransposedSizes() const noexcept
		{
			std::vector<std::pair<size_t, size_t>> ret;
			for (const auto& layer: _layers)
				ret.emplace_back(layer->GetNumberOfOutputs(), layer->GetNumberOfInputs());
			return ret;
		}
		
		void Evaluate(const Matrix& input, const bool needGradient, Matrix* const output = nullptr) const noexcept
		{
			// use input for first layer
			_layers[0]->Evaluate(input, true);
			
			// use previous activations for all other layers
			const size_t nLayers = GetSize();
			for (size_t l = 1; l < nLayers - 1; ++l)
				_layers[l]->Evaluate(this->_layers[l - 1]->GetActivation(), true);
			
			_layers[nLayers - 1]->Evaluate(this->_layers[nLayers - 2]->GetActivation(), needGradient, output);
		}
		
		double EvaluateTotalWeightCost() const noexcept
		{
			double weightCost = 0.0;
			for (const auto& layer: _layers)
			{
				const auto norm = layer->GetWeight().EuclideanNorm();
				weightCost += static_cast<double>(norm * norm);
			}
			
			return weightCost;
		}
		
		std::ostream& operator <<(std::ostream& stream) const noexcept override
		{
			stream << _layers.size() << std::endl;
			for (const auto& layer: _layers)
				*layer << stream;
			
			return stream;
		}
		
		std::istream& operator >>(std::istream& stream) noexcept override
		{
			_layers.clear();
			
			std::string line;
			
			std::getline(stream, line);
			const size_t nLayers = static_cast<size_t>(std::atoi(line.c_str()));
			
			for (size_t l = 0; l < nLayers; ++l)
			{
				std::getline(stream, line);
				const LayerType type = GetLayerType(line);
				
				std::getline(stream, line);
				const size_t nInput = static_cast<size_t>(std::atoi(line.c_str()));
				
				std::getline(stream, line);
				const size_t nOutput = static_cast<size_t>(std::atoi(line.c_str()));
				
				std::getline(stream, line);
				const ActivationFunctionType activationFunctionType = GetActivationFunctionType(line);
				auto activationFunction = ActivationFunctionFactory<mathDomain>::Create(activationFunctionType);
				
				auto layer = LayerFactory<mathDomain>::Create(type, nInput, nOutput, std::move(activationFunction),
				                                              std::move(TrivialBiasWeightInitializer<mathDomain>()));
				*layer >> stream;
				
				_layers.emplace_back(std::move(layer));
			}
			
			return stream;
		}
		
		using ConstIterator = typename Layers::const_iterator;
		ConstIterator begin() const noexcept { return _layers.begin(); }
		ConstIterator end() const noexcept { return _layers.end(); }
		const Layer& front() const noexcept { return _layers.front(); }
		const Layer& back() const noexcept { return _layers.back(); }
		const Layer& operator[](const size_t i) const noexcept { return _layers[i]; }
	
	protected:
		Layers _layers;
	};
}
