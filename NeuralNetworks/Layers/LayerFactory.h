#pragma once

#include <NeuralNetworks/Layers/All.h>
#include <NeuralNetworks/Layers/LayerType.h>

namespace nn
{
	template<MathDomain mathDomain>
	class LayerFactory
	{
	public:
		static std::unique_ptr<ILayer<mathDomain>> Create(const std::string& string)
		{
			return Create(GetActivationFunctionType(string));
		}
		
		template<typename... Args>
		static std::unique_ptr<ILayer<mathDomain>> Create(const LayerType type, Args&&... args)
		{
			std::unique_ptr<ILayer<mathDomain>> ret;
			switch (type)
			{
				case LayerType::Dense:
					ret = std::make_unique<DenseLayer<mathDomain>>(std::forward<Args>(args)...);
					break;
				case LayerType::SoftMax:
					ret = std::make_unique<SoftMaxLayer<mathDomain>>(std::forward<Args>(args)...);
					break;
				default:
					return nullptr;
			}
			
			assert(ret->GetType() == type);
			return ret;
		}
	};
}
