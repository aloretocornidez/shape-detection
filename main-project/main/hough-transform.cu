#include "hough-transform.hpp"
#include <cuda_runtime.h>
#include <iostream>

__global__ void houghTransform(char input[], char mask[], char output[], int rows, int cols, int maskWidth)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int pixVal = 0;

    if (row < rows && col < cols)
    {
        int startCol = col - maskWidth / 2;
        int startRow = row - maskWidth / 2;

        for (int j = 0; j < maskWidth; j++)
        {
            for (int k = 0; k < maskWidth; k++)
            {
                int curRow = startRow + j;
                int curCol = startCol + k;

                if (curRow > -1 && curRow < rows && curCol > -1 && curCol < cols)
                {
                    pixVal += input[curRow * cols + curCol] * mask[j * maskWidth + k];
                }
            }
        }
    }

    output[row * cols + col] = pixVal;
}

__global__ void addKernel(int size, int *input1, int *input2)
{

    int thread = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread < size)
    {
        input1[thread] = input1[thread] + input2[thread];
    }
}

void addKernelWrapper(int numberOfElements, int *input1, int *input2)
{

    std::cout << "Running Kernel Wrapper" << std::endl;

    std::cout << "Printing Input data passed to kernel setup input1 and input2" << std::endl;
    for (int i = 0; i < numberOfElements; i++)
    {
        std::cout << input1[i] << " and " << input2[i] << std::endl;
    }

    int *gpuInput1;
    int *gpuInput2;

    /* allocate memory on device, check for failure */
    if (cudaMalloc((void **)&gpuInput1, numberOfElements * sizeof(int)) != cudaSuccess)
    {
        std::cout << "malloc error for gpuInput1" << std::endl;
    }
    if (cudaMalloc((void **)&gpuInput2, numberOfElements * sizeof(int)) != cudaSuccess)
    {
        std::cout << "malloc error for gpuInput2" << std::endl;
    }

    /* copy data to device, check for failure, free device if needed */
    if (cudaMemcpy(gpuInput1, input1, numberOfElements * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(gpuInput1);
        cudaFree(gpuInput2);
        std::cout << "data transfer error from host to device on input1" << std::endl;
    }
    if (cudaMemcpy(gpuInput2, input2, numberOfElements * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(gpuInput1);
        cudaFree(gpuInput2);
        std::cout << "data transfer error from host to device on input2" << std::endl;
    }

    std::cout << "Printing Memory copied to gpu input 1 and 2" << std::endl;
    for (int i = 0; i < numberOfElements; i++)
    {
        std::cout << gpuInput1[i] << " and " << gpuInput2[i] << std::endl;
    }

    dim3 mygrid(ceil(numberOfElements / 256.0));
    dim3 myblock(256);

    addKernel<<<mygrid, myblock>>>(numberOfElements, input1, input2);

    /* copy data to host, check for failure, free device if needed */
    if (cudaMemcpy(input1, gpuInput1, numberOfElements * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cudaFree(gpuInput1);
        cudaFree(gpuInput2);
        printf("data transfer error from device to host on input1\n");
    }
    if (cudaMemcpy(input2, gpuInput2, numberOfElements * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cudaFree(gpuInput1);
        cudaFree(gpuInput2);
        printf("data transfer error from device to host on input2\n");
    }

    std::cout << "Finished Kernel Wrapper execution" << std::endl;
}