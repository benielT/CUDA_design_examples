
#pragma once

#define MAX_GRID_SIZE 256
#define MAX_BLOCK_SIZE 256

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <reduction_cpu.h>
#include <reduction_vanilla.cuh>
#include <reduction_thread_linear_addressing.cuh>
#include <reduction_shared_memory.cuh>
#include <reduction_shared_linear.cuh>
#include <reduction_warp_unroll.cuh>
#include <reduction_full_unroll.cuh>