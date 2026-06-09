#include <stdio.h>
#include <cuda_runtime.h>
#include "cooperative_groups.h"
namespace cg = cooperative_groups;

template <int T> __device__ void show_tile(const char *tag, cg::thread_block_tile<T> p) {
    int rank = p.thread_rank(); // thread rank in tile
    int size = p.size(); // number of threads in tile
    int mrank = p.meta_group_rank(); // rank fo the tile in parent 
    int msize = p.meta_group_size(); // number of tiles in parent

    printf("%s rank in the tile: %2d, size: %2d, rank in parent: %3d, num tile in parent: %3d, net size: %d\n", 
        tag, rank, size, mrank, msize, msize*size);
}

__global__ void cgwarp(int id) {

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (tid == 0 && bid == 0) {
        printf("Hello from Block %d, Thread %d!\n", bid, tid);
    }

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    auto warp32  = cg::tiled_partition<32>(block);  // 32 thread warps
    auto warp16  = cg::tiled_partition<16>(block); // 16 thread tiles
    auto warp8  = cg::tiled_partition<8>(block); // 8 thread tiles
    auto tile8 = cg::tiled_partition<8>(warp32); // 8 thread sub-warp
    auto tile4 = cg::tiled_partition<4>(tile8); // 4 thread sup-warp

    if (grid.thread_rank() == id ) {
        printf("warps and subwarps for thread %d\n", id);
        show_tile<32>("warp32", warp32);
        show_tile<16>("warp16", warp16);
        show_tile<8>("warp8", warp8);
        show_tile<8>("tile8", tile8);
        show_tile<4>("tile4", tile4);
    }
}

int main(int argc, char * argv[])
{
    int id = (argc > 1) ? atoi(argv[1]) : 12345;
    int blocks = (argc > 2) ? atoi(argv[2]) : 28800;
    int threads = (argc > 3) ? atoi(argv[3]) : 256;
    printf("Starting coog example\n");
    cgwarp<<< blocks, threads >>>(id);

    cudaError_t err = cudaDeviceSynchronize();
    
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    return 0;
}