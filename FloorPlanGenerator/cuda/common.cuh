#ifndef CUDA_COMMON
#define CUDA_COMMON

#include <stdint.h>

// Sorry, had to do it this way to make the reduce the cuda kernel registers usage
#define check_adjacency(a_up, a_down, a_left, a_right, b_up, b_down, b_left, b_right) (((a_down == b_up || a_up == b_down) && ((a_right > b_left && a_right <= b_right) || (a_left < b_right && a_left >= b_left) || (a_left <= b_left && a_right >= b_right))) ||  ((a_left == b_right || a_right == b_left) && ((a_down > b_up && a_down <= b_down) || (a_up < b_down && a_up >= b_up) || (a_up <= b_up && a_down >= b_down))))

__device__ inline uint8_t check_overlap(const int a_up, const int a_down, const int a_left, const int a_right, 
	const int b_up, const int b_down, const int b_left, const int b_right){
	if(((a_down > b_up && a_down <= b_down) ||
	(a_up  >= b_up && a_up < b_down)) &&
	((a_right > b_left && a_right <= b_right) ||
	(a_left  >= b_left && a_left  <  b_right) ||
	(a_left  <= b_left && a_right >= b_right))){
		return 0;
	}

	else if(((b_down > a_up && b_down <= a_down) ||
	(b_up >= a_up && b_up < a_down)) &&
	((b_right > a_left && b_right <= a_right) ||
	(b_left  >= a_left && b_left  <  a_right) ||
	(b_left  <= a_left && b_right >= a_right))){
		return 0;
	}

	else if(((a_right > b_left && a_right <= b_right) ||
	(a_left >= b_left && a_left < b_right)) &&
	((a_down > b_up && a_down <= b_down) ||
	(a_up  >= b_up && a_up   <  b_down) ||
	(a_up  <= b_up && a_down >= b_down))){
		return 0;
	}

	else if(((b_right > a_left && b_right <= a_right) ||
	(b_left >= a_left && b_left < a_right)) &&
	((b_down > a_up && b_down <= a_down) ||
	(b_up  >= a_up && b_up   <  a_down) ||
	(b_up  <= a_up && b_down >= a_down))){
		return 0;
	}

	return 1;
}

#endif