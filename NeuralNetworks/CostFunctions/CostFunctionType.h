#pragma once

namespace nn
{
	enum class CostFunctionType
	{
		__BEGIN__,
		
		Null = __BEGIN__,
		Quadratic,
		CrossEntropy,
		LogLikelihood,
		
		__END__
	};
	
	static inline std::string ToString(const CostFunctionType type) noexcept
	{
		switch (type)
		{
			case CostFunctionType::Quadratic:
				return "Quadratic";
			case CostFunctionType::CrossEntropy:
				return "Cross-Entropy";
			case CostFunctionType::LogLikelihood:
				return "Log-Likelihood";
			default:
				return "?";
		}
	}
	
	template<typename T>
	static inline CostFunctionType GetCostFunctionType(T&& string) noexcept
	{
		for (size_t type = 0; type < static_cast<size_t>(CostFunctionType::__END__); ++type)
		{
			auto t = static_cast<CostFunctionType>(type);
			if (string == ToString(t))
				return t;
		}
		
		return CostFunctionType::Null;
	}
}