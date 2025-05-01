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
#ifndef NO_SHARED
    /***********************************************************
     *                 3. Shared Memory Reduction              *            
     ***********************************************************/

    int device_id;
    cudaGetDevice(&device_id);

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    int shared_mem_per_thread = device_prop.sharedMemPerBlock / device_prop.maxThreadsPerBlock;

    std::cout << "*************************************************" << std::endl;
    std::cout << "*          Device Shared Memory Info            *" << std::endl;
    std::cout << "*************************************************" << std::endl;

    std::cout << "Device Name: " << device_prop.name << std::endl;
    std::cout << "Total Global Memory: " << device_prop.totalGlobalMem / (1 << 20) << " MB" << std::endl;
    std::cout << "Shared Memory Per Block: " << device_prop.sharedMemPerBlock / (1 << 10) << " KB" << std::endl;
    std::cout << "Max Threads Per Block: " << device_prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Per Multiprocessor: " << device_prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Shared Memory Per Thread: " << shared_mem_per_thread << " bytes" << std::endl;

    float total_time_shared = 0.0f;
    float result_shared = 0.0f;
    int shared_grid_size = (arr_size + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    thrust::device_vector<float> shared_block_sum(shared_grid_size);
    
    for (int i = 0; i < num_iterations; ++i)
    {
        thrust::sequence(dev_arr.begin(), dev_arr.end());
        thrust::fill(shared_block_sum.begin(), shared_block_sum.end(), 0.0f);
        auto start_shared = std::chrono::high_resolution_clock::now();
        result_shared = reduce_shared(dev_arr, shared_block_sum, arr_size);
        auto end_shared = std::chrono::high_resolution_clock::now();
        total_time_shared += std::chrono::duration<float, std::milli>(end_shared - start_shared).count();
    }

    float avg_time_shared = total_time_shared / num_iterations;

    // Calculate bandwidth
    float bandwidth_shared = (bytes_transferred / (avg_time_shared / 1000.0f)) / (1 << 30); // GB/s

    std::cout << "*************************************************" << std::endl;
    std::cout << "*                 Shared Memory Reduction       *" << std::endl;
    std::cout << "*************************************************" << std::endl;

    std::cout << "Result: " << result_shared << std::endl;
    std::cout << "Array Size: " << arr_size << std::endl;
    std::cout << "Average Runtime: " << avg_time_shared << " ms" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth_shared << " GB/s" << std::endl;
#endif 
#ifndef NO_SHARED_LINEAR
    /***********************************************************
     *           4. Shared Memory + Linear Reduction           *            
     ***********************************************************/

    float total_time_shared_linear = 0.0f;
    float result_shared_linear = 0.0f;
    thrust::device_vector<float> shared_linear_block_sum(MAX_GRID_SIZE);

    for (int i = 0; i < num_iterations; ++i)
    {
        thrust::sequence(dev_arr.begin(), dev_arr.end());
        thrust::fill(shared_linear_block_sum.begin(), shared_linear_block_sum.end(), 0.0f);
        auto start_shared_linear = std::chrono::high_resolution_clock::now();
        result_shared_linear = reduce_shared_linear(dev_arr, shared_linear_block_sum, arr_size);
        auto end_shared_linear = std::chrono::high_resolution_clock::now();
        total_time_shared_linear += std::chrono::duration<float, std::milli>(end_shared_linear - start_shared_linear).count();
    }

    float avg_time_shared_linear = total_time_shared_linear / num_iterations;

    // Calculate bandwidth
    float bandwidth_shared_linear = (bytes_transferred / (avg_time_shared_linear / 1000.0f)) / (1 << 30); // GB/s

    std::cout << "*************************************************" << std::endl;
    std::cout << "*           Shared Memory + Linear Reduction    *" << std::endl;
    std::cout << "*************************************************" << std::endl;

    std::cout << "Result: " << result_shared_linear << std::endl;
    std::cout << "Array Size: " << arr_size << std::endl;
    std::cout << "Average Runtime: " << avg_time_shared_linear << " ms" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth_shared_linear << " GB/s" << std::endl;
#endif
#ifndef NO_WARP_UNROLL
    float total_time_warp_unroll = 0.0f;
    float result_warp_unroll = 0.0f;
    thrust::device_vector<float> warp_unroll_block_sum(MAX_GRID_SIZE);

    for (int i = 0; i < num_iterations; ++i)
    {
        thrust::sequence(dev_arr.begin(), dev_arr.end());
        thrust::fill(warp_unroll_block_sum.begin(), warp_unroll_block_sum.end(), 0.0f);
        auto start_warp_unroll = std::chrono::high_resolution_clock::now();
        result_warp_unroll = reduce_warp_unroll(dev_arr, warp_unroll_block_sum, arr_size);
        auto end_warp_unroll = std::chrono::high_resolution_clock::now();
        total_time_warp_unroll += std::chrono::duration<float, std::milli>(end_warp_unroll - start_warp_unroll).count();
    }

    float avg_time_warp_unroll = total_time_warp_unroll / num_iterations;

    // Calculate bandwidth
    float bandwidth_warp_unroll = (bytes_transferred / (avg_time_warp_unroll / 1000.0f)) / (1 << 30); // GB/s

    std::cout << "*************************************************" << std::endl;
    std::cout << "*           Warp Unrolling Reduction            *" << std::endl;
    std::cout << "*************************************************" << std::endl;

    std::cout << "Result: " << result_warp_unroll << std::endl;
    std::cout << "Array Size: " << arr_size << std::endl;
    std::cout << "Average Runtime: " << avg_time_warp_unroll << " ms" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth_warp_unroll << " GB/s" << std::endl;
#endif

    return 0;
}