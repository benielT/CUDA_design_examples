#pragma once
#include <thrust/device_vector.h>

float reduce_shared(thrust::device_vector<float> dev_arr, thrust::device_vector<float> block_par_sum, int arr_size);