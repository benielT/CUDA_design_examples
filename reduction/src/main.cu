#include <iostream>
#include <thrust/sequence.h>
#include <reduction_kernels.h>

#include <chrono>
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{
    int arr_size = 1 << 24;
    thrust::device_vector<float> dev_arr(arr_size);
    thrust::host_vector<float> host_arr(arr_size);
    thrust::sequence(dev_arr.begin(), dev_arr.end());

    const int num_iterations = 10000;
    size_t bytes_transferred = arr_size * sizeof(float);

#ifndef NO_CPU
    /***********************************************************
     *                   0. CPU Reduction                      *            
     ***********************************************************/
    float total_time_cpu = 0.0f;
    float result_cpu = 0.0f;
    for (int i = 0; i < num_iterations; ++i)
    {
        thrust::sequence(host_arr.begin(), host_arr.end());
        auto start_cpu = std::chrono::high_resolution_clock::now();
        result_cpu = reduce_cpu(host_arr, arr_size);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        total_time_cpu += std::chrono::duration<float, std::milli>(end_cpu - start_cpu).count();
    }
    float avg_time_cpu = total_time_cpu / num_iterations;
    // Calculate bandwidth
    float bandwidth_cpu = (bytes_transferred / (avg_time_cpu / 1000.0f)) / (1 << 30); // GB/s
    std::cout << "*************************************************" << std::endl;
    std::cout << "*                   CPU Reduction               *" << std::endl;
    std::cout << "*************************************************" << std::endl;
    std::cout << "Result: " << result_cpu << std::endl;
    std::cout << "Array Size: " << arr_size << std::endl;
    std::cout << "Average Runtime: " << avg_time_cpu << " ms" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth_cpu << " GB/s" << std::endl;
#endif

#ifndef NO_VANILLA
    /***********************************************************
     *                 1. Vanilla Reduction                    *            
     ***********************************************************/
    float total_time_vanilla = 0.0f;
    float result_vanilla = 0.0f;

    for (int i = 0; i < num_iterations; ++i)
    {
        thrust::sequence(dev_arr.begin(), dev_arr.end());
        auto start_vanilla = std::chrono::high_resolution_clock::now();
        result_vanilla = reduce_vanilla(dev_arr, arr_size);
        auto end_vanilla = std::chrono::high_resolution_clock::now();
        total_time_vanilla += std::chrono::duration<float, std::milli>(end_vanilla - start_vanilla).count();
    }

    float avg_time_vanilla = total_time_vanilla / num_iterations;

    // Calculate bandwidth
    float bandwidth_vanilla = (bytes_transferred / (avg_time_vanilla / 1000.0f)) / (1 << 30); // GB/s

    std::cout << "*************************************************" << std::endl;
    std::cout << "*                 Vanilla Reduction             *" << std::endl;
    std::cout << "*************************************************" << std::endl;

    std::cout << "Result: " << result_vanilla << std::endl;
    std::cout << "Array Size: " << arr_size << std::endl;
    std::cout << "Average Runtime: " << avg_time_vanilla << " ms" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth_vanilla << " GB/s" << std::endl;
#endif
#ifndef NO_THREAD_LINEAR
    /***********************************************************
     *           2. Thread Linear Addressing Reduction         *            
     ***********************************************************/

    float total_time_thread_linear = 0.0f;
    float result_thread_linear = 0.0f;

    for (int i = 0; i < num_iterations; ++i)
    {
        thrust::sequence(dev_arr.begin(), dev_arr.end());
        auto start_thread_linear = std::chrono::high_resolution_clock::now();
        result_thread_linear = reduce_vanilla(dev_arr, arr_size);
        auto end_thread_linear = std::chrono::high_resolution_clock::now();
        total_time_thread_linear += std::chrono::duration<float, std::milli>(end_thread_linear - start_thread_linear).count();
    }

    float avg_time_thread_linear = total_time_thread_linear / num_iterations;

    // Calculate bandwidth
    float bandwidth_thread_linear = (bytes_transferred / (avg_time_thread_linear / 1000.0f)) / (1 << 30); // GB/s

    std::cout << "*************************************************" << std::endl;
    std::cout << "*           Thread Linear Addressing            *" << std::endl;
    std::cout << "*************************************************" << std::endl;

    std::cout << "Result: " << result_thread_linear << std::endl;
    std::cout << "Array Size: " << arr_size << std::endl;
    std::cout << "Average Runtime: " << avg_time_thread_linear << " ms" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth_thread_linear << " GB/s" << std::endl;
#endif

    return 0;
}