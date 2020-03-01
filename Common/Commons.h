#pragma once

inline float random0to1() {
	return (float)rand() / (float)RAND_MAX;
}

inline int divUp(int a, int b) {
	if (b == 0) {
		return 1;
	}
	int result = (a % b != 0) ? (a / b + 1) : (a / b);
	return result;
}