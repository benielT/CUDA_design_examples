#include <thrust/device_vector.h>

void reduce_shared_linear(float* dev_arr, float* block_sum, int arr_size);