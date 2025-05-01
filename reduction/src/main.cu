#include <iostream>
#include <thrust/sequence.h>
#include <reduction_kernels.h>

#include <chrono>
#include <cuda_runtime.h>
#include <fstream>

int main(int argc, char *argv[])
{
    int arr_size = 1 << 22;
    thrust::device_vector<float> dev_arr(arr_size);
    thrust::host_vector<float> host_arr(arr_size);
    thrust::sequence(dev_arr.begin(), dev_arr.end());

    const int num_iterations = 1000;
    size_t bytes_transferred = arr_size * sizeof(float);

    std::string profile_filename = "profile_summary.csv";

    std::ofstream fstream;
    fstream.open(profile_filename, std::ios::out | std::ios::trunc);

    if (!fstream.is_open()) {
        std::cerr << "Error: Could not open the file " << profile_filename << std::endl;
        return 1; // Indicate an error occurred
    }

    fstream << "Reduction Type,Array Size,Average Runtime (ms),Achieved Bandwidth (GB/s)\n";
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
    float bandwidth_cpu = (bytes_transferred / (avg_time_cpu / 1000.0f)) / (1 << 30); // GB/s
    std::cout << "*************************************************" << std::endl;
    std::cout << "*                   CPU Reduction               *" << std::endl;
    std::cout << "*************************************************" << std::endl;
    std::cout << "Result: " << result_cpu << std::endl;
    std::cout << "Array Size: " << arr_size << std::endl;
    std::cout << "Average Runtime: " << avg_time_cpu << " ms" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth_cpu << " GB/s" << std::endl;
    fstream << "CPU," << arr_size << "," << avg_time_cpu << "," << bandwidth_cpu << "\n";
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
        auto dev_arr_ptr = thrust::raw_pointer_cast(dev_arr.data());
        auto start_vanilla = std::chrono::high_resolution_clock::now();
        reduce_vanilla(dev_arr_ptr, arr_size);
        auto end_vanilla = std::chrono::high_resolution_clock::now();
        result_vanilla = dev_arr[0];
        total_time_vanilla += std::chrono::duration<float, std::milli>(end_vanilla - start_vanilla).count();
    }

    float avg_time_vanilla = total_time_vanilla / num_iterations;
    float bandwidth_vanilla = (bytes_transferred / (avg_time_vanilla / 1000.0f)) / (1 << 30); // GB/s

    std::cout << "*************************************************" << std::endl;
    std::cout << "*                 Vanilla Reduction             *" << std::endl;
    std::cout << "*************************************************" << std::endl;

    std::cout << "Result: " << result_vanilla << std::endl;
    std::cout << "Array Size: " << arr_size << std::endl;
    std::cout << "Average Runtime: " << avg_time_vanilla << " ms" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth_vanilla << " GB/s" << std::endl;
    fstream << "Vanilla," << arr_size << "," << avg_time_vanilla << "," << bandwidth_vanilla << "\n";
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
        auto dev_arr_ptr = thrust::raw_pointer_cast(dev_arr.data());
        auto start_thread_linear = std::chrono::high_resolution_clock::now();
        reduce_thread_linear(dev_arr_ptr, arr_size);
        auto end_thread_linear = std::chrono::high_resolution_clock::now();
        result_thread_linear = dev_arr[0];
        total_time_thread_linear += std::chrono::duration<float, std::milli>(end_thread_linear - start_thread_linear).count();
    }

    float avg_time_thread_linear = total_time_thread_linear / num_iterations;
    float bandwidth_thread_linear = (bytes_transferred / (avg_time_thread_linear / 1000.0f)) / (1 << 30); // GB/s

    std::cout << "*************************************************" << std::endl;
    std::cout << "*           Thread Linear Addressing            *" << std::endl;
    std::cout << "*************************************************" << std::endl;

    std::cout << "Result: " << result_thread_linear << std::endl;
    std::cout << "Array Size: " << arr_size << std::endl;
    std::cout << "Average Runtime: " << avg_time_thread_linear << " ms" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth_thread_linear << " GB/s" << std::endl;
    fstream << "Thread Linear," << arr_size << "," << avg_time_thread_linear << "," << bandwidth_thread_linear << "\n";
#endif

#ifndef NO_SHARED
    /***********************************************************
     *                 3. Shared Memory Reduction              *            
     ***********************************************************/
    float total_time_shared = 0.0f;
    float result_shared = 0.0f;
    thrust::device_vector<float> shared_block_sum((arr_size + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE);

    for (int i = 0; i < num_iterations; ++i)
    {
        thrust::sequence(dev_arr.begin(), dev_arr.end());
        thrust::fill(shared_block_sum.begin(), shared_block_sum.end(), 0.0f);
        auto dev_arr_ptr = thrust::raw_pointer_cast(dev_arr.data());
        auto shared_block_sum_ptr = thrust::raw_pointer_cast(shared_block_sum.data());
        auto start_shared = std::chrono::high_resolution_clock::now();
        reduce_shared(dev_arr_ptr, shared_block_sum_ptr, arr_size);
        auto end_shared = std::chrono::high_resolution_clock::now();
        result_shared = dev_arr[0];
        total_time_shared += std::chrono::duration<float, std::milli>(end_shared - start_shared).count();
    }

    float avg_time_shared = total_time_shared / num_iterations;
    float bandwidth_shared = (bytes_transferred / (avg_time_shared / 1000.0f)) / (1 << 30); // GB/s

    std::cout << "*************************************************" << std::endl;
    std::cout << "*                 Shared Memory Reduction       *" << std::endl;
    std::cout << "*************************************************" << std::endl;

    std::cout << "Result: " << result_shared << std::endl;
    std::cout << "Array Size: " << arr_size << std::endl;
    std::cout << "Average Runtime: " << avg_time_shared << " ms" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth_shared << " GB/s" << std::endl;
    fstream << "Shared Memory," << arr_size << "," << avg_time_shared << "," << bandwidth_shared << "\n";
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
        auto dev_arr_ptr = thrust::raw_pointer_cast(dev_arr.data());
        auto shared_linear_block_sum_ptr = thrust::raw_pointer_cast(shared_linear_block_sum.data());
        auto start_shared_linear = std::chrono::high_resolution_clock::now();
        reduce_shared_linear(dev_arr_ptr, shared_linear_block_sum_ptr, arr_size);
        auto end_shared_linear = std::chrono::high_resolution_clock::now();
        result_shared_linear = dev_arr[0];
        total_time_shared_linear += std::chrono::duration<float, std::milli>(end_shared_linear - start_shared_linear).count();
    }

    float avg_time_shared_linear = total_time_shared_linear / num_iterations;
    float bandwidth_shared_linear = (bytes_transferred / (avg_time_shared_linear / 1000.0f)) / (1 << 30); // GB/s

    std::cout << "*************************************************" << std::endl;
    std::cout << "*           Shared Memory + Linear Reduction    *" << std::endl;
    std::cout << "*************************************************" << std::endl;

    std::cout << "Result: " << result_shared_linear << std::endl;
    std::cout << "Array Size: " << arr_size << std::endl;
    std::cout << "Average Runtime: " << avg_time_shared_linear << " ms" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth_shared_linear << " GB/s" << std::endl;
    fstream << "Shared + Linear," << arr_size << "," << avg_time_shared_linear << "," << bandwidth_shared_linear << "\n";
#endif

#ifndef NO_WARP_UNROLL
    /***********************************************************
     *           5. Warp Unrolling Reduction                  *            
     ***********************************************************/
    float total_time_warp_unroll = 0.0f;
    float result_warp_unroll = 0.0f;
    thrust::device_vector<float> warp_unroll_block_sum(MAX_GRID_SIZE);

    for (int i = 0; i < num_iterations; ++i)
    {
        thrust::sequence(dev_arr.begin(), dev_arr.end());
        thrust::fill(warp_unroll_block_sum.begin(), warp_unroll_block_sum.end(), 0.0f);
        auto dev_arr_ptr = thrust::raw_pointer_cast(dev_arr.data());
        auto warp_unroll_block_sum_ptr = thrust::raw_pointer_cast(warp_unroll_block_sum.data());
        auto start_warp_unroll = std::chrono::high_resolution_clock::now();
        reduce_warp_unroll(dev_arr_ptr, warp_unroll_block_sum_ptr, arr_size);
        auto end_warp_unroll = std::chrono::high_resolution_clock::now();
        result_warp_unroll = dev_arr[0];
        total_time_warp_unroll += std::chrono::duration<float, std::milli>(end_warp_unroll - start_warp_unroll).count();
    }

    float avg_time_warp_unroll = total_time_warp_unroll / num_iterations;
    float bandwidth_warp_unroll = (bytes_transferred / (avg_time_warp_unroll / 1000.0f)) / (1 << 30); // GB/s

    std::cout << "*************************************************" << std::endl;
    std::cout << "*           Warp Unrolling Reduction            *" << std::endl;
    std::cout << "*************************************************" << std::endl;

    std::cout << "Result: " << result_warp_unroll << std::endl;
    std::cout << "Array Size: " << arr_size << std::endl;
    std::cout << "Average Runtime: " << avg_time_warp_unroll << " ms" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth_warp_unroll << " GB/s" << std::endl;
    fstream << "Shared + Linear + Warp Unroll," << arr_size << "," << avg_time_warp_unroll << "," << bandwidth_warp_unroll << "\n";
#endif

#ifndef NO_FULL_UNROLL
    /***********************************************************
     *           6. Full Unrolling Reduction                  *            
     ***********************************************************/
    float total_time_full_unroll = 0.0f;
    float result_full_unroll = 0.0f;    
    thrust::device_vector<float> full_unroll_block_sum(MAX_GRID_SIZE);
    for (int i = 0; i < num_iterations; ++i)
    {
        thrust::sequence(dev_arr.begin(), dev_arr.end());
        thrust::fill(full_unroll_block_sum.begin(), full_unroll_block_sum.end(), 0.0f);
        auto dev_arr_ptr = thrust::raw_pointer_cast(dev_arr.data());
        auto full_unroll_block_sum_ptr = thrust::raw_pointer_cast(full_unroll_block_sum.data());
        auto start_full_unroll = std::chrono::high_resolution_clock::now();
        reduce_full_unroll(dev_arr_ptr, full_unroll_block_sum_ptr, arr_size);
        auto end_full_unroll = std::chrono::high_resolution_clock::now();
        result_full_unroll = dev_arr[0];
        total_time_full_unroll += std::chrono::duration<float, std::milli>(end_full_unroll - start_full_unroll).count();
    }
    float avg_time_full_unroll = total_time_full_unroll / num_iterations;
    float bandwidth_full_unroll = (bytes_transferred / (avg_time_full_unroll / 1000.0f)) / (1 << 30); // GB/s
    std::cout << "*************************************************" << std::endl;
    std::cout << "*           Full Unrolling Reduction             *" << std::endl;
    std::cout << "*************************************************" << std::endl;
    std::cout << "Result: " << result_full_unroll << std::endl;
    std::cout << "Array Size: " << arr_size << std::endl;
    std::cout << "Average Runtime: " << avg_time_full_unroll << " ms" << std::endl;
    std::cout << "Achieved Bandwidth: " << bandwidth_full_unroll << " GB/s" << std::endl;
    fstream << "Shared + Linear + Full Unroll," << arr_size << "," << avg_time_full_unroll << "," << bandwidth_full_unroll << "\n";
#endif

    fstream.close();

    if (fstream.good()) { // Check if operations were successful after closing
        std::cout << "Successfully wrote data to " << profile_filename << std::endl;
    } else {
        std::cerr << "Error occurred during writing to " << profile_filename << std::endl;
        return 1; // Indicate an error occurred
    }
    return 0;
}