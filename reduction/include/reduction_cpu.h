#pragma once
#include <thrust/host_vector.h>

float reduce_cpu(thrust::host_vector<float> arr, int arr_size);