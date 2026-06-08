
#pragma once

#define MAX_GRID_SIZE 512
#define MAX_BLOCK_SIZE 512

#include <cuda_runtime.h>
#include "cooperative_groups.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <reduction_cpu.h>
#include <reduction_vanilla.cuh>
#include <reduction_thread_linear_addressing.cuh>
#include <reduction_shared_memory.cuh>
#include <reduction_shared_linear.cuh>
#include <reduction_warp_unroll.cuh>
#include <reduction_full_unroll.cuh>
#include <reduction_full_unroll_syncwarp.cuh>
#include <reduction_full_unroll_warpshufl.cuh>
#include <reduction_warp_unroll_warpshufl_atomicadd.cuh>