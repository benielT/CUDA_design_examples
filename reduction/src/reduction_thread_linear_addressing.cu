#include <reduction_kernels.h>

__global__ void reduce_thread_linear_kernel(float *arr, int arr_size)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    float thread_sum = 0.0f;

    for (int i = thread_id; i < arr_size; i += blockDim.x * gridDim.x)
    {
        thread_sum += arr[i];
    }

    arr[thread_id] = thread_sum;
}

float reduce_thread_linear(thrust::device_vector<float> dev_arr, int arr_size)
{
    reduce_thread_linear_kernel<<<MAX_GRID_SIZE, MAX_BLOCK_SIZE>>>(dev_arr.data().get(), arr_size);
    reduce_thread_linear_kernel<<<1, MAX_BLOCK_SIZE>>>(dev_arr.data().get(), MAX_BLOCK_SIZE*MAX_GRID_SIZE);
    reduce_thread_linear_kernel<<<1, 1>>>(dev_arr.data().get(), MAX_BLOCK_SIZE);
    float result = dev_arr[0];
    return result;
}