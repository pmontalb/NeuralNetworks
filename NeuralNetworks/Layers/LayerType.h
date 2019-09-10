#pragma once

namespace nn
{
	enum class LayerType
	{
		__BEGIN__,
		Null = __BEGIN__,
		Dense,
		SoftMax,
		
		__END__
	};
	
	static inline std::string ToString(const LayerType type) noexcept
	{
		switch (type)
		{
			case LayerType::Dense:
				return "Dense";
			case LayerType::SoftMax:
				return "SoftMax";
			default:
				return "?";
		}
	}
	
	template<typename T>
	static inline LayerType GetLayerType(T&& string) noexcept
	{
		for (size_t type = 0; type < static_cast<size_t>(LayerType::__END__); ++type)
		{
			auto t = static_cast<LayerType>(type);
			if (string == ToString(t))
				return t;
		}
		
		return LayerType::Null;
	}
}
