#include <thrust/device_vector.h>

void reduce_warp_unroll(float* dev_arr, float* block_sum, int arr_size);