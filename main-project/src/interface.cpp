#include "interface.hpp"
#include "hough-transform.hpp"
#include <iostream>


void CUDAtestAddingKernel()
{

  int *input1;
  int *input2;

  int size = 1024;

  input1 = (int *)malloc(size * sizeof(int));
  input2 = (int *)malloc(size * sizeof(int));

  for (int i = 0; i < size; i++)
  {
    input1[i] = i;
    input2[i] = i;
  }

  // Call the GPU execution kernel.
  cudaAddKernel(size, input1, input2);

  free(input1);
  free(input2);
}
