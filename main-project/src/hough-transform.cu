#include "hough-transform.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

// Thread Block and Memory Allocation Parameters.
#define BLOCK_DIMENSIONS 32

// Hough Tranform Kernel Definitions
#define HOUGH_TRANSFORM_NAIVE_KERNEL 1
#define DEGREES_TO_RADIANS 0.0120830485

// Sample Kernel for memory access of an image.
#if 0
__global__ void set_image_to_value(uchar *inputImage, int height, int width)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int column = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < height && column < width)
    {

        inputImage[row * width + column] = 128;
    }
}
#endif

void cpuKernelHoughTransform(cv::Mat &srcImage, std::vector<cv::Vec3f> &srcCircles, int distance, int minimumRadius, int maximumRadius)
{
    std::cout << "Executing the Hough Transform on the CPU." << std::endl;

    if (minimumRadius < 0)
    {
        std::cerr << "Minimum radius must be 1 or greater." << std::endl;
        exit(-1);
    }
    if (minimumRadius == 0)
    {
        minimumRadius = 5;
    }
    if (maximumRadius == 0)
    {
        maximumRadius = min(srcImage.rows - 1, srcImage.cols - 1) / 2;
    }

    /* Begin Algoritm */
    for (int radius = minimumRadius; radius < maximumRadius; radius++)
    {
        // Note, the threshold for the number of pixels is dynamically set within the loop.

        int threshold = ((log(radius * 2 / 3)) * 80) / log(3);
        // std::cout << "Testing Radius: " << radius << " | With Threshold: " << threshold << std::endl;

        for (int row = radius; row < srcImage.rows - radius; row += distance)
        {
            for (int column = radius; column < srcImage.cols - radius; column += distance)
            {
                // std::cout << "Testing (row, column): (" << row << ", " << column << ")" << std::endl;
                int accumulator = 0;
                // Check if the a circle exists at the coordinate point (with the current radius)
                for (int theta = 0; theta < 360; theta++)
                {
                    // Checking all 4 cardinal directions.
                    int x;
                    int y;
                    int deltaX = cos(theta * DEGREES_TO_RADIANS) * radius;
                    int deltaY = sin(theta * DEGREES_TO_RADIANS) * radius;
                    x = deltaX + column;
                    y = deltaY + row;

                    if (srcImage.at<uchar>(x, y) < 10)
                    {
                        accumulator++;
                    }
                }

                // Adding the coordinate if the contained enough edge pixels.
                if (accumulator > threshold)
                {
                    // std::cout << "Accumulator was greater than threshold. :(" << accumulator << ", " << threshold << ")" << std::endl;
                    // printf("Adding a set of circle parameters: [%d, %d, %d]\n\n", row, column, radius);

                    srcCircles.push_back({(float)row, (float)column, (float)radius});
                }
            }
        }
    }

    std::cout << "Execution of the Hough Transform on the CPU comleted." << std::endl;
}

// This kernel uses the global memory to write to the R-table.
// Each thread will test
__global__ void hough_transform_kernel_naive(uchar *srcImage, unsigned int *rTable, int imageRows, int imageColumns, int minimumRadius, int maximumRadius, int distance)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int column = threadIdx.x + blockIdx.x * blockDim.x;
    int numberOfRadii = maximumRadius - minimumRadius;

    // DEBUG STATEMENTS
    // minimumRadius = 50;
    // maximumRadius = 55;

    // Check if the thread is within image bounds.
    if (row < (imageRows - minimumRadius) && row > minimumRadius && column < (imageColumns - minimumRadius) && column > minimumRadius)
    {

        for (int radius = 0; radius < maximumRadius - minimumRadius; radius++)
        {
            for (int theta = 0; theta < 360; theta++)
            {
                int deltaX = cos(DEGREES_TO_RADIANS * theta) * radius;
                int deltaY = sin(DEGREES_TO_RADIANS * theta) * radius;

                int imageIndex = (row + deltaY) * imageColumns + (row + deltaX);

                int pixelValue = srcImage[0];

                if (pixelValue < 10)
                {

                    atomicAdd(&rTable[0], 1);
                }
            }
        }
    }
}

void houghTransform(cv::Mat &srcImage, std::vector<cv::Vec3f> &circles, int method)
{
    /* Parameters Until we make them modular */
    int distance = 1;
    int minimumRadius = 18;  // 20 is the min for the test image.
    int maximumRadius = 100; // 100 is max radius for the test image.
    int imageRows = srcImage.rows;
    int imageColumns = srcImage.cols;

    // Run Hough Transform on the CPU and return.
    if (method == 0)
    {
        std::cout << "Executing Hough Transform on the CPU" << std::endl;
        cpuKernelHoughTransform(srcImage, circles, distance, minimumRadius, maximumRadius);
        return;
    }

    // Allocate GPU Memory Initialize pointer for the GPU memory
    uchar *gpuImageBuffer;
    cudaError_t err = cudaMalloc((void **)&gpuImageBuffer, imageRows * imageColumns * sizeof(uchar));
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Allocating an r-table to populate parameters for each shape.
    // Allocate the R table on the GPU.
    unsigned int *deviceRTable;
    err = cudaMalloc((void **)&deviceRTable, imageColumns * imageRows * (maximumRadius - minimumRadius) * sizeof(unsigned int));
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Copy Data from host to Device
    err = cudaMemcpy(gpuImageBuffer, srcImage.ptr<uchar>(0, 0), imageRows * imageColumns * sizeof(uchar), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    std::cout << "Cuda Memory Allocated." << std::endl;

    /*
     *
     * Execute the hough transform on the specified kernel, populating the accumulator table.
     *
     */
    if (method == HOUGH_TRANSFORM_NAIVE_KERNEL)
    {
        std::cout << "Executing Hough Transform on the Naive Kernel" << std::endl;

        dim3 mygrid(ceil(imageColumns / (BLOCK_DIMENSIONS * 1.0)), ceil(imageRows / (BLOCK_DIMENSIONS * 1.0)));
        dim3 myblock(BLOCK_DIMENSIONS, BLOCK_DIMENSIONS);

        hough_transform_kernel_naive<<<mygrid, myblock>>>(gpuImageBuffer, deviceRTable, imageRows, imageColumns, minimumRadius, maximumRadius, distance);

        std::cout << "Hough Transform Naive Kernel Execution Complete" << std::endl;
    }
    else
    {
        std::cout << "Invalid Kernel Method Chosen. | (method): " << method << std::endl;
    }

    /* Modications are not being made to the image, so no copy back to the host is required. */
    // Copy data from device to host
    unsigned int *hostRTable;
    hostRTable = (unsigned int *)malloc(imageColumns * imageRows * (maximumRadius - minimumRadius) * sizeof(unsigned int));
    err = cudaMemcpy(hostRTable, deviceRTable, imageColumns * imageRows * (maximumRadius - minimumRadius) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cudaFree(gpuImageBuffer);
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    parseRTable(circles, hostRTable, minimumRadius, maximumRadius, imageRows, imageColumns);

    // Free allocated memory
    cudaFree(deviceRTable);
    cudaFree(gpuImageBuffer);
    free(hostRTable);
    std::cout << "Cuda Memory Freed" << std::endl;

    std::cout << "GPU Hough Transform Execution Complete" << std::endl;
}

void parseRTable(std::vector<cv::Vec3f> &circles, unsigned int *rTable, int minimumRadius, int maximumRadius, int imageRows, int imageColumns)
{

    for (int row = 0; row < imageRows - minimumRadius; row++)
    {
        for (int column = 0; column < imageColumns - minimumRadius; column++)
        {
            for (int radius = 0; radius < (maximumRadius - minimumRadius); radius++)
            {
                // Check if the image at that coordinate is greater than the threshold.
                // If so, then append the circle to the circles vector.
                if (row == 50 && column == 50 && radius == 20)
                {
                    circles.push_back({(float)row, (float)column, (float)radius});
                }
            }
        }
    }
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
}
