#pragma once

#include <Optimizers/MiniBatchData.h>
#include <NeuralNetworks/Stopwatch.h>
#include <NeuralNetworks/Layers/Initializers/IBiasWeightInitializer.h>
#include <NeuralNetworks/Layers/ILayer.h>

#include <NeuralNetworks/ISerializable.h>
#include <NeuralNetworks/Activations/ActivationFunctionFactory.h>
#include <NeuralNetworks/Layers/LayerFactory.h>
#include <NeuralNetworks/Layers/Initializers/TrivialBiasWeightInitializer.h>

namespace nn
{
	template<MathDomain mathDomain> class IOptimizer;
	
	template<MathDomain mathDomain>
	class Network: public ISerializable
	{
		using mat = Matrix<mathDomain>;
		using vec = Vector<mathDomain>;
		using lay = std::unique_ptr<ILayer<mathDomain>>;
	public:
		explicit Network(std::vector<lay>&& layers) noexcept;
		
		explicit Network(std::istream& stream) noexcept;
		
		void Evaluate(mat& out, const mat& in, const int debugLevel = 0) const noexcept;
		
		void Train(IOptimizer<mathDomain>& optimizer, const NetworkTrainingData<mathDomain>& networkTrainingData) noexcept;
		
		std::ostream& operator <<(std::ostream& stream) const noexcept override;
		std::istream& operator >>(std::istream& stream) noexcept override;
		
		inline const std::vector<lay>& GetLayers() const noexcept { return _layers; }
		inline auto GetNumberOfLayers() const noexcept { return GetLayers().size(); }
		
	private:
		std::vector<lay> _layers {};
	};
}

#include <NeuralNetworks/Network.tpp>
