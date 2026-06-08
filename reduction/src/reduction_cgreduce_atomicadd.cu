#include <reduction_kernels.h>
#include <cuda/atomic>
//include/reduction_kernels.h contains #include <cooperative_groups.h>, #include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void reduce_cgreduce_atomicadd_kernel(const float * __restrict arr, float * __restrict block_sum, int arr_size) {
    // extern __shared__ float shared_arr[];

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
    // shared_arr[local_id] = thread_sum;

    warp.sync();

    // warp unroll is replaced with cooperative group reduce operation
    thread_sum = cg::reduce(warp, thread_sum, cg::plus<float>());
        
    if (warp.thread_rank() == 0) 
        atomicAdd_block(&block_sum[block.group_index().x], thread_sum);
}

void reduce_cgreduce_atomicadd(float* dev_arr, float *block_sum, int arr_size)
{
    reduce_cgreduce_atomicadd_kernel<<<MAX_GRID_SIZE, MAX_BLOCK_SIZE>>>(
         dev_arr,block_sum,arr_size);

    reduce_cgreduce_atomicadd_kernel<<<1, MAX_GRID_SIZE>>>(
        block_sum, dev_arr, MAX_GRID_SIZE);
    cudaDeviceSynchronize();
}
