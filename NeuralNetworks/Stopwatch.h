#pragma once

#include <chrono>

namespace nn
{
	class Stopwatch
	{
	public:
		explicit Stopwatch(bool startTimer = true)
		{
			if (startTimer)
				Start();
		}
		
		//#define DO_NOT_USE_TIMER
		#ifndef DO_NOT_USE_TIMER
			inline void Start() noexcept { start = std::chrono::high_resolution_clock::now(); }
			inline void Stop() noexcept { stop = std::chrono::high_resolution_clock::now(); }
			auto GetNanoSeconds() const noexcept { return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count(); }
		#else
			inline void Start() const noexcept { }
			inline void Stop() const noexcept { }
			auto GetNanoSeconds() const noexcept { return 0.0; }
		#endif
		
		auto GetMicroSeconds() const noexcept { return static_cast<double>(GetNanoSeconds()) * 1e-3; }
		auto GetMilliSeconds() const noexcept { return static_cast<double>(GetNanoSeconds()) * 1e-6; }
		auto GetSeconds() const noexcept { return static_cast<double>(GetNanoSeconds()) * 1e-9; }
		
	private:
		std::chrono::high_resolution_clock::time_point start {};
		std::chrono::high_resolution_clock::time_point stop {};
	};
	
}
