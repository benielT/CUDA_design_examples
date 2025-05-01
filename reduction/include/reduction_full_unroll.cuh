#include <thrust/device_vector.h>

float reduce_full_unroll(thrust::device_vector<float> dev_arr, thrust::device_vector<float> block_sum, int arr_size);