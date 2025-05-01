#include <reduction_kernels.h>

__global__ void reduce_shared_linear_kernel(float *arr, float *block_sum, int arr_size)
{
    extern __shared__ float shared_arr[];
    int local_id = threadIdx.x;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float thread_sum = 0.0f;

    // Initialize shared memory + linear addressing
    for (int i = thread_id; i < arr_size; i += stride) {
        thread_sum += arr[i];
    }
    shared_arr[local_id] = thread_sum;

    __syncthreads();

    // Reduction in shared memory
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (local_id < i) {
            shared_arr[local_id] += shared_arr[local_id + i];
        }
        __syncthreads();
    }

    if (local_id == 0) {
        block_sum[blockIdx.x] = shared_arr[0];
    }
}

void reduce_shared_linear(float* dev_arr, float* block_sum, int arr_size)
{
    reduce_shared_linear_kernel<<<MAX_GRID_SIZE, MAX_BLOCK_SIZE, MAX_BLOCK_SIZE * sizeof(float)>>>(
        dev_arr,block_sum,arr_size);

    reduce_shared_linear_kernel<<<1, MAX_GRID_SIZE, MAX_GRID_SIZE * sizeof(float)>>>(
        block_sum, dev_arr, MAX_GRID_SIZE);
    cudaDeviceSynchronize();
}
