
#include <iostream>
#include <thrust/sequence.h>
#include <reduction_kernels.h>

#include <chrono>
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{
    int arr_size = 1 << 20;
    thrust::device_vector<float> dev_arr(arr_size);
    thrust::sequence(dev_arr.begin(), dev_arr.end());

    auto start = std::chrono::high_resolution_clock::now();
    float result = reduce_vanilla(dev_arr, arr_size);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;

    // Calculate bandwidth
    size_t bytes_transferred = arr_size * sizeof(float);
    float bandwidth = (bytes_transferred / (duration.count() / 1000.0f)) / (1 << 30); // GB/s

    // // Get device properties
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // float peak_bandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / (1 << 30); // GB/s

    std::cout << "Result: " << result << std::endl;
    std::cout << "Array Size: " << arr_size << std::endl;
    std::cout << "Runtime: " << duration.count() << " ms" << std::endl;
    // std::cout << "Peak Bandwidth: " << peak_bandwidth << " GB/s" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth << " GB/s" << std::endl;
    // std::cout << "Percentage of Peak Bandwidth: " << (bandwidth / peak_bandwidth) * 100.0f << " %" << std::endl;

    return 0;
}