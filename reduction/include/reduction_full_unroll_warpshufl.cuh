#pragma once

#include "cooperative_groups.h"

void reduce_full_unroll_warpshufl(float* dev_arr, float* block_sum, int arr_size);