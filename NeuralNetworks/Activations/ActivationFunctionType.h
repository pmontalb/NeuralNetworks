#pragma once

namespace nn
{
	enum class ActivationFunctionType
	{
		__BEGIN__,
		Null = __BEGIN__,
		BentIdentity,
		
		ExponentialLinearUnity,
		Elu = ExponentialLinearUnity,
		
		HyperbolicTangent,
		Tanh = HyperbolicTangent,
		
		InverseSquareRootLinearUnit,
		IsrLu = InverseSquareRootLinearUnit,
		
		LeakyRectifiedLinearUnit,
		LeakyReLu = LeakyRectifiedLinearUnit,
		
		RectifiedLinearUnit,
		ReLu = RectifiedLinearUnit,
		
		Sigmoid,
		SoftMax,
		
		__END__
	};
	
	static inline std::string ToString(const ActivationFunctionType type) noexcept
	{
		switch (type)
		{
			case ActivationFunctionType::BentIdentity:
				return "BentIdentity";
			case ActivationFunctionType::ExponentialLinearUnity:
				return "ExponentialLinearUnit";
			case ActivationFunctionType::HyperbolicTangent:
				return "HyperbolicTangent";
			case ActivationFunctionType::InverseSquareRootLinearUnit:
				return "InverseSquareRootLinearUnit";
			case ActivationFunctionType::LeakyRectifiedLinearUnit:
				return "LeakyRectifiedLinearUnit";
			case ActivationFunctionType::RectifiedLinearUnit:
				return "RectifiedLinearUnit";
			case ActivationFunctionType::Sigmoid:
				return "Sigmoid";
			case ActivationFunctionType::SoftMax:
				return "SoftMax";
			default:
				return "?";
		}
	}
	
	template<typename T>
	static inline ActivationFunctionType GetActivationFunctionType(T&& string) noexcept
	{
		for (size_t type = 0; type < static_cast<size_t>(ActivationFunctionType::__END__); ++type)
		{
			auto t = static_cast<ActivationFunctionType>(type);
			if (string == ToString(t))
				return t;
		}
		
		return ActivationFunctionType::Null;
	}
	
}
