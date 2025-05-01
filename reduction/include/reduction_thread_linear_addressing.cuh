#pragma once

#include <thrust/device_vector.h>

void reduce_thread_linear(float* dev_arr, int arr_size);