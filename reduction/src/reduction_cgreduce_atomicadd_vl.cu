#include <reduction_kernels.h>
#include <cuda/atomic>
//include/reduction_kernels.h contains #include <cooperative_groups.h>, #include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

inline __device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void reduce_cgreduce_atomicadd_kernel_vl(const float  * __restrict arr, float * __restrict block_sum, int arr_size) {
    // extern __shared__ float shared_arr[];

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    int local_id = block.thread_rank(); // rank in the block 
    int thread_id = grid.thread_rank(); // rank in terms of universal thread id
    int stride = grid.size();
    // the temporary variable is declared as vector type of 4xfloat 
    float4 thread_sum_v4 = {0.0f, 0.0f, 0.0f, 0.0f};
    int adj_array_size = arr_size >> 2; // the folding takes less iterations
    
    // Initialize shared memory + linear addressing
    for (int i = thread_id; i < adj_array_size; i += stride) {
	// vector addition not work directly. so added operation overload helper above
        thread_sum_v4 = thread_sum_v4 + reinterpret_cast<const float4 *>(arr)[i];
    }
    
    // Adding final vectorized partial_sums into single partial sum
    float thread_sum = thread_sum_v4.x + thread_sum_v4.y 
	    + thread_sum_v4.z + thread_sum_v4.w;
    // shared_arr[local_id] = thread_sum;
    warp.sync();

    // warp unroll is replaced with cooperative group reduce operation
    thread_sum = cg::reduce(warp, thread_sum, cg::plus<float>());
        
    if (warp.thread_rank() == 0) 
        atomicAdd_block(&block_sum[block.group_index().x], thread_sum);
}

void reduce_cgreduce_atomicadd_vl(float* dev_arr, float *block_sum, int arr_size)
{
    reduce_cgreduce_atomicadd_kernel_vl<<<MAX_GRID_SIZE, MAX_BLOCK_SIZE>>>(
         dev_arr,block_sum,arr_size);

    reduce_cgreduce_atomicadd_kernel_vl<<<1, MAX_GRID_SIZE>>>(
        block_sum, dev_arr, MAX_GRID_SIZE);
    cudaDeviceSynchronize();
}
