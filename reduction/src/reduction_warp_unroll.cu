#include <reduction_kernels.h>

__device__ inline void warp_reduce(volatile float *sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
__global__ void reduce_warp_unroll_kernel(float *arr, float *block_sum, int arr_size)
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
    for (int i = blockDim.x / 2; i > 32; i >>= 1) {
        if (local_id < i) {
            shared_arr[local_id] += shared_arr[local_id + i];
        }
        __syncthreads();
    }

    if (local_id < 32)
        warp_reduce(shared_arr, local_id);
    
    if (local_id == 0) {
        block_sum[blockIdx.x] = shared_arr[0];
    }
}

float reduce_warp_unroll(thrust::device_vector<float> dev_arr, thrust::device_vector<float> block_sum, int arr_size)
{

    reduce_warp_unroll_kernel<<<MAX_GRID_SIZE, MAX_BLOCK_SIZE, MAX_BLOCK_SIZE * sizeof(float)>>>(
        dev_arr.data().get(),block_sum.data().get(),arr_size);

    reduce_warp_unroll_kernel<<<1, MAX_GRID_SIZE, MAX_GRID_SIZE * sizeof(float)>>>(
        block_sum.data().get(), dev_arr.data().get(), MAX_GRID_SIZE);
    cudaDeviceSynchronize();

    float result = dev_arr[0];
    return result;
}
