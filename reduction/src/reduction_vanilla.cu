#include <reduction_kernels.h>

__global__ void reduce_vanilla_kernel(float *arr, int m)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    arr[thread_id] += arr[thread_id + m];
}

float reduce_vanilla(thrust::device_vector<float> dev_arr, int arr_size)
{
    for (int i = arr_size / 2; i > 0; i /= 2)
    {
        int threads = std::min(MAX_BLOCK_SIZE, i);
        int blocks = std::max(i/MAX_BLOCK_SIZE, 1);

        reduce_vanilla_kernel<<<blocks, threads>>>(dev_arr.data().get(), i);
    }
    cudaDeviceSynchronize();
    float result = dev_arr[0];
    return result;
}