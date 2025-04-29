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

    const int num_iterations = 100;

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
    size_t bytes_transferred = arr_size * sizeof(float);
    float bandwidth_vanilla = (bytes_transferred / (avg_time_vanilla / 1000.0f)) / (1 << 30); // GB/s

    std::cout << "*************************************************" << std::endl;
    std::cout << "*                 Vanilla Reduction             *" << std::endl;
    std::cout << "*************************************************" << std::endl;

    std::cout << "Result: " << result_vanilla << std::endl;
    std::cout << "Array Size: " << arr_size << std::endl;
    std::cout << "Average Runtime: " << avg_time_vanilla << " ms" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth_vanilla << " GB/s" << std::endl;


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

    return 0;
}