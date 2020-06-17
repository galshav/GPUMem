#include <windows.h>
#include <cinttypes>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>

typedef const char* const* const const_char_array_t;
enum class RETURN_CODE
{
	SUCCESS =  0,
	FAILURE = -1,
	SEH		=  2,
};
enum class COPY_DIRECTION
{
	TO_GPU = cudaMemcpyHostToDevice,
	TO_CPU = cudaMemcpyDeviceToHost,
};

/*
	GPU memory wrapper.
	Ownership of given buffer is taken.
	The host buffer would not be available for use after
	instantiation of the class.
	The caller of the Get method is responsible to free
	the allocated host buffer.
*/
template<typename T>
class GPUMemory
{
public:
	GPUMemory(const T* buffer, const std::size_t size) :
		m_DeviceBuffer(nullptr),
		m_BufferSize(size)
	{
		// Allocating device buffer on the GPU.
		const auto allocationResult = cudaMalloc<T>((T**)&m_DeviceBuffer, size);
		
		// Copy data from host to device GPU.
		const auto memcpyResult = cudaMemcpy(
			(void*)m_DeviceBuffer, 
			(void*)buffer, 
			size, 
			static_cast<cudaMemcpyKind>(COPY_DIRECTION::TO_GPU));
		
		// Remove data from host.
		ZeroMemory((void*)buffer, size);
	}

	T* Get(void)
	{
		T* ptr = new T[m_BufferSize];
		const auto memcpyResult = cudaMemcpy(
			(void*)ptr,
			(void*)m_DeviceBuffer,
			m_BufferSize,
			static_cast<cudaMemcpyKind>(COPY_DIRECTION::TO_CPU));
		return ptr;
	}

	~GPUMemory() noexcept
	{
		cudaFree(m_DeviceBuffer);
	}

private:
	T* m_DeviceBuffer = nullptr;
	std::size_t m_BufferSize = 0;
};

__host__ int main(const std::uint8_t argc, const_char_array_t argv)
{
	UNREFERENCED_PARAMETER(argc);
	UNREFERENCED_PARAMETER(argv);

	std::vector<int> buffer = { 0xff };

	// Initiate host buffer with some data.
	const std::uint8_t hostBuffer[] =
	{0xde, 0xad, 0xbe, 0xef,
	 0xde, 0xad, 0xbe, 0xef,
	 0xde, 0xad, 0xbe, 0xef,
	 0xde, 0xad, 0xbe, 0xef,
	 0xde, 0xad, 0xbe, 0xef};

	// Send buffer to GPU for later use.
	GPUMemory<std::uint8_t> gpuBuffer(hostBuffer, sizeof(hostBuffer));

	/*
	...
	hostBuffer is not available here.
	Can not be found in process memory dump.
	...
	*/

	// Use the hidden GPU memory only when required.
	const auto hiddenBuffer = gpuBuffer.Get();

	delete hiddenBuffer;
	return static_cast<int>(RETURN_CODE::SUCCESS);
}