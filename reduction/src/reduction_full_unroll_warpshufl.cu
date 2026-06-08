#include <reduction_kernels.h>
//include/reduction_kernels.h contains #include "cooperative_groups.h"
namespace cg = cooperative_groups;

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
__global__ void reduce_full_unroll_warpshufl_kernel(float *arr, float *block_sum, int arr_size)
{
    extern __shared__ float shared_arr[];

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    int local_id = block.thread_rank(); // rank in the block 
    int thread_id = grid.thread_rank(); // rank in terms of universal thread id
    int stride = grid.size();
    float thread_sum = 0.0f;

    // Initialize shared memory + linear addressing
    for (int i = thread_id; i < arr_size; i += stride) {
        thread_sum += arr[i];
    }
    shared_arr[local_id] = thread_sum;

    block.sync(); //only the threads in the block need to be sync this will allow unwanded disturbance to other blocks
    
    if (blockSize > 512 && local_id < 512 && local_id + 512 < blockSize) {
          shared_arr[local_id] += shared_arr[local_id + 512];
    }
    block.sync();
    if (blockSize > 256 && local_id < 256 && local_id + 256 < blockSize) {
          shared_arr[local_id] += shared_arr[local_id + 256];
    }
    block.sync();
    if (blockSize > 128 && local_id < 128 && local_id + 128 < blockSize) {
          shared_arr[local_id] += shared_arr[local_id + 128];
    }
    block.sync();

    if (blockSize > 64 && local_id < 64 && local_id + 64 < blockSize) {
          shared_arr[local_id] += shared_arr[local_id + 64];
    }
    block.sync();

    /* This means for example for 256 block, will have 8 warps and the 
        first warp will have meta_group rank 0 */
    if (warp.meta_group_rank() == 0){ 
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
        shared_arr[local_id] += shared_arr[local_id + 32]; warp.sync();
        shared_arr[local_id] += warp.shfl_down(shared_arr[local_id], 16);
        shared_arr[local_id] += warp.shfl_down(shared_arr[local_id], 8);
        shared_arr[local_id] += warp.shfl_down(shared_arr[local_id], 4);
        shared_arr[local_id] += warp.shfl_down(shared_arr[local_id], 2);
        shared_arr[local_id] += warp.shfl_down(shared_arr[local_id], 1);
        if (local_id == 0) block_sum[blockIdx.x] = shared_arr[0];
// #else
        // warp_reduce<blockSize>(shared_arr, local_id);
// #endif
    }
    
    // if (local_id == 0) {
    //     block_sum[blockIdx.x] = shared_arr[0];
    // }
}

void reduce_full_unroll_warpshufl(float* dev_arr, float *block_sum, int arr_size)
{
    reduce_full_unroll_warpshufl_kernel<MAX_BLOCK_SIZE><<<MAX_GRID_SIZE, MAX_BLOCK_SIZE, MAX_BLOCK_SIZE * sizeof(float)>>>(
        dev_arr,block_sum,arr_size);

    reduce_full_unroll_warpshufl_kernel<MAX_GRID_SIZE><<<1, MAX_GRID_SIZE, MAX_GRID_SIZE * sizeof(float)>>>(
        block_sum, dev_arr, MAX_GRID_SIZE);
    cudaDeviceSynchronize();
}
