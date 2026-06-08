#include <reduction_kernels.h>

template <unsigned int blockSize>
__device__ inline void warp_reduce(volatile float *sdata, int tid)
{
    if (blockSize >= 64){
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
    else if (blockSize >= 32){
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
    else if (blockSize >= 16){
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
        
    }
    else if (blockSize >= 8){
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
    else if (blockSize >= 4){
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
    else if (blockSize >= 2){
        sdata[tid] += sdata[tid + 1];
    }
}

template <unsigned int blockSize>
__global__ void reduce_full_unroll_syncwarp_kernel(float *arr, float *block_sum, int arr_size)
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
    // for (int i = blockDim.x / 2; i > 32; i >>= 1) {
    //     if (local_id < i) {
    //         shared_arr[local_id] += shared_arr[local_id + i];
    //     }
    //     __syncthreads();
    // }
    if (blockSize >= 512){
        if (local_id < 256) {  
          shared_arr[local_id] += shared_arr[local_id + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256){
        if (local_id < 128) {  
          shared_arr[local_id] += shared_arr[local_id + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128){
        if (local_id < 64) {  
          shared_arr[local_id] += shared_arr[local_id + 64];
        }
        __syncthreads();
    }

    if (local_id < 32){
        // Instead of using custom warp unroll using __syncwarp();
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
                            shared_arr[local_id] += shared_arr[local_id + 32]; __syncwarp();
        if (local_id < 16 ) shared_arr[local_id] += shared_arr[local_id + 16]; __syncwarp();
        if (local_id < 8 )  shared_arr[local_id] += shared_arr[local_id + 8]; __syncwarp();
        if (local_id < 4 )  shared_arr[local_id] += shared_arr[local_id + 4]; __syncwarp();
        if (local_id < 2 )  shared_arr[local_id] += shared_arr[local_id + 2]; __syncwarp();
        if (local_id < 1 )  shared_arr[local_id] += shared_arr[local_id + 1]; __syncwarp();
// #else
        // warp_reduce<blockSize>(shared_arr, local_id);
// #endif
    }
    
    if (local_id == 0) {
        block_sum[blockIdx.x] = shared_arr[0];
    }
}

void reduce_full_unroll_syncwarp(float* dev_arr, float *block_sum, int arr_size)
{
    reduce_full_unroll_syncwarp_kernel<MAX_BLOCK_SIZE><<<MAX_GRID_SIZE, MAX_BLOCK_SIZE, MAX_BLOCK_SIZE * sizeof(float)>>>(
        dev_arr,block_sum,arr_size);

    reduce_full_unroll_syncwarp_kernel<MAX_GRID_SIZE><<<1, MAX_GRID_SIZE, MAX_GRID_SIZE * sizeof(float)>>>(
        block_sum, dev_arr, MAX_GRID_SIZE);
    cudaDeviceSynchronize();
}
