#ifndef __hough_transform__
#define __hough_transform__

#include <cuda_runtime.h>

__global__ void houghTransform(char input[], char mask[], char output[], int rows, int cols, int maskWidth);

__global__ void addKernel(int kernelSize, int* input1, int* input2, int* output);

void addKernelWrapper(int size, int* input1, int* input2);

#endif