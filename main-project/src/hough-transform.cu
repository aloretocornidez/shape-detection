#include "hough-transform.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void hough_transform_kernel_naive(uchar *inputImage, int height, int width)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int column = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < height && column < width)
    {
        // printf("%d\n", inputImage[row * width + column]);

        inputImage[row * width + column] = 128;

        // printf("%d\n", inputImage[row * width + column]);
    }

}

void cudaHoughTransform(cv::Mat &grayscaleInputImage, cv::InputArray circles)
{

    std::cout << "Running CUDA hough transform." << std::endl;

    int imageRows = grayscaleInputImage.rows;
    int imageColumns = grayscaleInputImage.cols;


    // Initializing pointers for the GPU memory
    uchar *gpuImageBuffer;

    // Error detection
    cudaError_t err;

    // Allocate GPU Memory
    err = cudaMalloc((void **)&gpuImageBuffer, imageRows * imageColumns * sizeof(uchar));
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }



    // Copy Data from host to Device
    err = cudaMemcpy(gpuImageBuffer, grayscaleInputImage.ptr<uchar>(0, 0), imageRows * imageColumns * sizeof(uchar), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    // std::cout << "Cuda Memory Allocated." << std::endl;

    // Execute kernel
    const int threads = 32;
    dim3 mygrid(ceil(imageColumns / (threads * 1.0)), ceil(imageRows / (threads * 1.0)));
    dim3 myblock(threads, threads);

    std::cout << "Executing kernel." << std::endl;
    hough_transform_kernel_naive<<<mygrid, myblock>>>(gpuImageBuffer, imageRows, imageColumns);
    std::cout << "Kernel execution complete." << std::endl;

    // Copy data from device to host
    std::cout << "Copying from device to host." << std::endl;
    err = cudaMemcpy(grayscaleInputImage.ptr<uchar>(0, 0), gpuImageBuffer, imageRows * imageColumns * sizeof(uchar), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cudaFree(gpuImageBuffer);
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    std::cout << "Memory Copied." << std::endl;

    cudaDeviceSynchronize();

    // for (int row = 0; row < grayscaleInputImage.rows; row++)
    // {
    //     for (int column = 0; column < grayscaleInputImage.cols; column++)
    //     {
    //         std::cout << (int)grayscaleInputImage.at<uchar>(row, column) << std::endl;
    //         // grayscaleInputImage.at<uchar>(row, column) = 128; // column % 256;
    //     }
    // }

    // Free allocated memory
    cudaFree(gpuImageBuffer);



    std::cout << "Finished CUDA hough transform." << std::endl;
}

__global__ void add_kernel_basic(int size, int *input1, int *input2)
{

    int thread = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread < size)
    {
        input1[thread] = input1[thread] + input2[thread];
    }
}

void cudaAddKernel(int array_size, int *array_1, int *array_2)
{

    std::cout << "Running Kernel Wrapper" << std::endl;

    // Initializing pointers to the gpu memory
    int *gpu_array_1;
    int *gput_array_2;

    // allocate memory on device, check for failure
    if (cudaMalloc((void **)&gpu_array_1, array_size * sizeof(int)) != cudaSuccess)
    {
        std::cout << "malloc error for gpuInput1" << std::endl;
    }
    if (cudaMalloc((void **)&gput_array_2, array_size * sizeof(int)) != cudaSuccess)
    {
        std::cout << "malloc error for gpuInput2" << std::endl;
    }

    // copy data to device, check for failure, free device if needed

    cudaError_t err; // Use this whenever calling cudaMalloc and cudaMemcpy.

    err = cudaMemcpy(gpu_array_1, array_1, array_size * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(gput_array_2, array_2, array_size * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Execute kernel.
    dim3 mygrid(ceil(array_size / 256.0));
    dim3 myblock(256);

    add_kernel_basic<<<mygrid, myblock>>>(array_size, gpu_array_1, gput_array_2);

    // copy data to host, check for failure, free device if needed
    if (cudaMemcpy(array_1, gpu_array_1, array_size * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cudaFree(gpu_array_1);
        cudaFree(gput_array_2);
        printf("data transfer error from device to host on input1\n");
    }
    if (cudaMemcpy(array_2, gput_array_2, array_size * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cudaFree(gpu_array_1);
        cudaFree(gput_array_2);
        printf("data transfer error from device to host on input2\n");
    }

    cudaFree(gpu_array_1);
    cudaFree(gput_array_2);

    std::cout << "Finished Kernel Wrapper execution" << std::endl;
}
