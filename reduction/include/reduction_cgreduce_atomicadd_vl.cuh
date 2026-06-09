#pragma once

#include "cooperative_groups.h"
#include <cuda/atomic>

void reduce_cgreduce_atomicadd_vl(float* dev_arr, float *block_sum, int arr_size);
