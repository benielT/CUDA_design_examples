#pragma once
#include <thrust/device_vector.h>

void reduce_shared(float* dev_arr, float* block_par_sum, int arr_size);