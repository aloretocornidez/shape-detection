#include<stdlib.h>
#include<time.h>
#include<stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>
// #define CUDA_CHECK(ans)                                                   \
//   { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line,
//                       bool abort = true) {
//   if (code != cudaSuccess) {
//     fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
//             file, line);
//     if (abort)
//       exit(code);
//   }
// }

cudaError_t err;

void serialConvolution(unsigned char input[], unsigned char mask[], unsigned char output[], int rows, int cols, int maskWidth);

//NAIVE 2D CONVOLUTION
__global__ void naiveConvolution(unsigned char input[], unsigned char mask[], unsigned char output[], int rows, int cols, int maskWidth){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int pixVal = 0;

    if(row < rows && col < cols){
        int startCol = col - maskWidth / 2;
        int startRow = row - maskWidth / 2;

        for(int j = 0; j < maskWidth; j++){
            for(int k = 0; k < maskWidth; k++){
                int curRow = startRow + j;
                int curCol = startCol + k;

                if(curRow > -1 && curRow < rows && curCol > -1 && curCol < cols){
                    pixVal += input[curRow * cols + curCol] * mask[j * maskWidth + k];
                }
            }
        }
    }

    output[row * cols + col] = pixVal;
}

//CONSTANT MEMORY 2D CONVOLUTION
//This is a sample using 5x5 constant memory. Uses the same input array as others
#define maskDimension 5
__constant__ unsigned char consMask[maskDimension*maskDimension];

__global__ void constantConvolution(unsigned char input[], unsigned char output[], int rows, int cols, int maskWidth){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int pixVal = 0;

    if(row < rows && col < cols){
        int startCol = col - maskWidth / 2;
        int startRow = row - maskWidth / 2;

        for(int j = 0; j < maskWidth; j++){
            for(int k = 0; k < maskWidth; k++){
                int curRow = startRow + j;
                int curCol = startCol + k;

                if(curRow > -1 && curRow < rows && curCol > -1 && curCol < cols){
                    pixVal += input[curRow * cols + curCol] * consMask[j * maskWidth + k];
                }
            }
        }
    }

    output[row * cols + col] = pixVal;
}

//TILED 2D CONVOLUTION
//Also uses constant memory
//This is done assuming we know the tile width/ block dimensions
#define convolution1BlockDim 32
__global__ void tiledConvolution(unsigned char input[], unsigned char output[], int rows, int cols, int maskWidth){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    //Assuming we know the blocm width, to use 2D notation
    __shared__ sharedM[convolution1BlockDim][convolution1BlockDim];

    outputSizeSquare = 28; //We will produce 28 pixel values for each 32 threads

    //For loop
    for(int outer = 0; outer < cols; )

    //Bring in all of the data


    if(row < rows && col < cols){
        int startCol = col - maskWidth / 2;
        int startRow = row - maskWidth / 2;

        for(int j = 0; j < maskWidth; j++){
            for(int k = 0; k < maskWidth; k++){
                int curRow = startRow + j;
                int curCol = startCol + k;

                if(curRow > -1 && curRow < rows && curCol > -1 && curCol < cols){
                    pixVal += input[curRow * cols + curCol] * consMask[j * maskWidth + k];
                }
            }
        }
    }

    output[row * cols + col] = pixVal;
}


//MAIN FUNCTION
int main(void){
    srand(time(NULL)); //Set up random values
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //We will compute a convolution image on a 720p made up image
    unsigned char inputImage[720][1280];
    unsigned char outputImage[720][1280];
    unsigned char mask3x3[3][3];
    unsigned char mask5x5[5][5];

    //Initialize input image 
    for(int i = 0; i < 720; i++){
        for(int j = 0; j < 1280; j++){
            inputImage[i][j] = abs(rand() % 10);
        }
    }

    //Initialize 3x3 mask and 5x5 mask
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            mask3x3[i][j] = abs(rand() % 10);
        }
    }
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            mask5x5[i][j] = abs(rand() % 10);
        }
    }

    //Apply convolution kernel to test values for accuracy
    //Will be using a 5x5 for now
    int kCols = 5;
    int kRows = 5;
    int kColDisplace = kCols / 2;
    int kRowDisplace = kRows / 2;

	std::cout << "Hello!" << std::endl;

    cudaEventRecord(start);
    for(int i=0; i < 720; ++i)              // rows
    {
        for(int j=0; j < 1280; ++j)          // columns
        {
            int startRow = i - kRowDisplace;
            int startCol = j - kColDisplace;
            unsigned char sum = 0;
            
            for(int m=0; m < kRows; ++m) { //Kernel rows
                for(int n=0; n < kCols; ++n) { //Kernel Cols
                    //int nn = kCols - 1 - n;  // column index of flipped kernel
                    int currRow = startRow + m;
                    int currCol = startCol + n;

                    if(currRow > -1 && currRow < 720 && currCol > -1 && currCol < 1280){
                        sum += inputImage[currRow][currCol] * mask5x5[m][n];
                    }
                }
            }
            outputImage[i][j] = sum;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed time for serial: " << milliseconds << std::endl;

    //Now we will do a naive implementation on CUDA
    unsigned char* hostInput;
    unsigned char* hostMask;
    unsigned char* hostOutput;
    unsigned char* deviceInput;
    unsigned char* deviceMask;
    unsigned char* deviceOutput;
    int rows = 720;
    int cols = 1280;
    int maskVal = 5;
    cudaEvent_t startNaive, stopNaive;
    cudaEventCreate(&startNaive);
    cudaEventCreate(&stopNaive);

    //Allocate Memory on host side
    hostInput = (unsigned char*)malloc(rows * cols * sizeof(unsigned char));
    hostMask = (unsigned char*)malloc(maskVal * maskVal * sizeof(unsigned char));
    hostOutput = (unsigned char*)malloc(rows * cols * sizeof(unsigned char));

    //Populare arrays on the host side
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            hostInput[i * cols + j] = inputImage[i][j];
        }
    }
    for(int i = 0; i < maskVal; i++){
        for(int j = 0; j < maskVal; j++){
            hostMask[i * maskVal + j] = mask5x5[i][j];
        }
    }

    //Allocate GPU memory here
    err = 
        cudaMalloc((void **)&deviceInput, rows * cols * sizeof(unsigned char));
    err = 
        cudaMalloc((void **)&deviceMask, maskVal * maskVal * sizeof(unsigned char));
    err= 
        cudaMalloc((void **)&deviceOutput, rows * cols * sizeof(unsigned char));
    cudaDeviceSynchronize();

    //Populate arrays on device side
    err = cudaMemcpy(deviceInput, hostInput,
                            rows * cols * sizeof(unsigned char),
                            cudaMemcpyHostToDevice);
    err = cudaMemcpy(deviceMask, hostMask,
                            maskVal * maskVal * sizeof(unsigned char),
                            cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    //Call the naive kernel
    cudaEventRecord(startNaive);
    int blockSize = 32;
    dim3 blockDim(blockSize,blockSize), gridDim( 1 + (cols - 1) / blockSize,
                                             1 + (rows - 1) / blockSize);
    naiveConvolution<<<gridDim, blockDim>>>
        (deviceInput, deviceMask, deviceOutput, rows, cols, maskVal);
    cudaEventRecord(stopNaive);
    cudaEventSynchronize(stopNaive);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startNaive, stopNaive);
    std::cout << "Elapsed time for naive: " << milliseconds <<std::endl;

    err = cudaMemcpy(hostOutput, deviceOutput,
                            rows * cols * sizeof(unsigned char),
                            cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //Check wether output is correct
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            assert(hostOutput[i * cols + j] == outputImage[i][j]);
        }
    }
    

    //CUDA CONSTANT MEM
    cudaMemcpyToSymbol(consMask, hostMask, maskVal * maskVal);
    cudaEvent_t startConstant, stopConstant;
    cudaEventCreate(&startConstant);
    cudaEventCreate(&stopConstant);

    cudaEventRecord(startConstant);
    constantConvolution<<<gridDim, blockDim>>>
        (deviceInput, deviceOutput, rows, cols, maskVal);
    cudaEventRecord(stopConstant);
    cudaEventSynchronize(stopConstant);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startConstant, stopConstant);
    std::cout << "Elapsed time for constant: " << milliseconds <<std::endl;
    


    //Memory Freeing
    //CUDA
    cudaFree(deviceInput);
    cudaFree(deviceMask);
    cudaFree(deviceOutput);
    

    //CPU
    free(hostInput);
    free(hostMask);
    free(hostOutput);
}