#pragma once

#include "cooperative_groups.h"
#include <cuda/atomic>

void reduce_warp_unroll_warpshufl_atomicadd(float* dev_arr, float *block_sum, int arr_size);