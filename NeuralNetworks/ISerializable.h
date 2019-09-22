#pragma once

namespace nn
{
	class ISerializable
	{
	public:
		ISerializable() noexcept = default;
		virtual ~ISerializable() noexcept = default;
		ISerializable(const ISerializable&) noexcept = default;
		ISerializable& operator=(const ISerializable&) noexcept = default;
		
		virtual std::ostream& operator <<(std::ostream& stream) const noexcept = 0;
		virtual std::istream& operator >>(std::istream& stream) noexcept = 0;
		
		void Serialize(std::ostream& stream) const noexcept { *this << stream; }
		void Deserialize(std::istream& stream) noexcept { *this >> stream; }
	};
}
