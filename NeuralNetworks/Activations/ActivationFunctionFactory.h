#pragma once

#include <NeuralNetworks/Activations/All.h>
#include <NeuralNetworks/Activations/ActivationFunctionType.h>

namespace nn
{
	template<MathDomain mathDomain>
	class ActivationFunctionFactory
	{
	public:
		template<typename... Args>
		static std::unique_ptr<IActivationFunction<mathDomain>> Create(const std::string& string, Args&&... args)
		{
			return Create(GetActivationFunctionType(string, std::forward<Args>(args)...));
		}
		
		template<typename... Args>
		static std::unique_ptr<IActivationFunction<mathDomain>> Create(const ActivationFunctionType type, Args&&... args)
		{
			std::unique_ptr<IActivationFunction<mathDomain>> ret;
			switch (type)
			{
				case ActivationFunctionType::BentIdentity:
					ret = std::make_unique<BentIdentityActivationFunction<mathDomain>>(std::forward<Args>(args)...);
					break;
				case ActivationFunctionType::ExponentialLinearUnity:
					ret = std::make_unique<EluActivationFunction<mathDomain>>(std::forward<Args>(args)...);
					break;
				case ActivationFunctionType::HyperbolicTangent:
					ret = std::make_unique<TanhActivationFunction<mathDomain>>(std::forward<Args>(args)...);
					break;
				case ActivationFunctionType::InverseSquareRootLinearUnit:
					ret = std::make_unique<IsrLuActivationFunction<mathDomain>>(std::forward<Args>(args)...);
					break;
				case ActivationFunctionType::LeakyRectifiedLinearUnit:
					ret = std::make_unique<LeakyReLuActivationFunction<mathDomain>>(std::forward<Args>(args)...);
					break;
				case ActivationFunctionType::RectifiedLinearUnit:
					ret = std::make_unique<ReLuActivationFunction<mathDomain>>(std::forward<Args>(args)...);
					break;
				case ActivationFunctionType::Sigmoid:
					ret = std::make_unique<SigmoidActivationFunction<mathDomain>>(std::forward<Args>(args)...);
					break;
				case ActivationFunctionType::SoftMax:
					ret = std::make_unique<SoftMaxActivationFunction<mathDomain>>(std::forward<Args>(args)...);
					break;
				default:
					return nullptr;
			}
			
			assert(ret->GetType() == type);
			return ret;
		}
	};
}
