#pragma once

#include <thrust/device_vector.h>

void reduce_full_unroll_syncwarp(float* dev_arr, float* block_sum, int arr_size);