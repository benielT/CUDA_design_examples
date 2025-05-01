#include <reduction_kernels.h>

__global__ void reduce_shared_kernel(float *arr, float *block_sum)
{
    extern __shared__ float shared_arr[];
    int local_id = threadIdx.x;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    shared_arr[local_id] = arr[thread_id];
  
    __syncthreads();

    for (unsigned int i = 1; i < blockDim.x; i *= 2) {
        if (local_id % (2 * i) == 0) {
            shared_arr[local_id] += shared_arr[local_id + i];
        }
        __syncthreads();
    }

    if (local_id == 0) {
        block_sum[blockIdx.x] = shared_arr[0];
    }
}

void reduce_shared(float* dev_arr, float* block_par_sum, int arr_size)
{
    int grid_size = (arr_size + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    // First reduction pass
    reduce_shared_kernel<<<grid_size, MAX_BLOCK_SIZE, MAX_BLOCK_SIZE * sizeof(float)>>>(
        dev_arr, block_par_sum);

    // Second reduction pass
    while (grid_size > MAX_BLOCK_SIZE) {
        int new_grid_size = (grid_size + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
        reduce_shared_kernel<<<new_grid_size, MAX_BLOCK_SIZE, MAX_BLOCK_SIZE * sizeof(float)>>>(
            block_par_sum, block_par_sum);
        grid_size = new_grid_size;
    }
    // Final reduction pass
    reduce_shared_kernel<<<1, grid_size, grid_size * sizeof(float)>>>(
        block_par_sum, dev_arr);
    cudaDeviceSynchronize();
}
