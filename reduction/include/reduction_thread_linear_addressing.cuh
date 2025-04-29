#pragma once

#include <thrust/device_vector.h>

float reduce_thread_linear(thrust::device_vector<float> dev_arr, int arr_size);