#include <reduction_kernels.h>

__global__ void reduce_thread_linear_kernel(float *arr, int arr_size)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float thread_sum = 0.0f;

    for (int i = thread_id; i < arr_size; i += stride)
    {
        thread_sum += arr[i];
    }

    arr[thread_id] = thread_sum;
}

void reduce_thread_linear(float* dev_arr, int arr_size)
{
    reduce_thread_linear_kernel<<<MAX_GRID_SIZE, MAX_BLOCK_SIZE>>>(dev_arr, arr_size);
    reduce_thread_linear_kernel<<<1, MAX_BLOCK_SIZE>>>(dev_arr, MAX_BLOCK_SIZE*MAX_GRID_SIZE);
    reduce_thread_linear_kernel<<<1, 1>>>(dev_arr, MAX_BLOCK_SIZE);
    cudaDeviceSynchronize();
}