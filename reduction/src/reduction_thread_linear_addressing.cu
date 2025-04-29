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
    __syncthreads();
}

float reduce_thread_linear(thrust::device_vector<float> dev_arr, int arr_size)
{
    for (int i = arr_size / 2; i > 0; i /= 2)
    {
        int threads = std::min(MAX_BLOCK_SIZE, i);
        int blocks = std::max(i/MAX_BLOCK_SIZE, 1);

        reduce_thread_linear_kernel<<<blocks, threads>>>(dev_arr.data().get(), i);
    }
    cudaDeviceSynchronize();
    float result = dev_arr[0];
    return result;
}