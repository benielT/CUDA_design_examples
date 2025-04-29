
#pragma once
#include <thrust/device_vector.h>

float reduce_vanilla(thrust::device_vector<float> dev_arr, int arr_size);