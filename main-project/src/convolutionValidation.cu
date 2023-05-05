#include<stdlib.h>
#include<time.h>
#include<stdio.h>
//#include <cuda_runtime.h>
#include<iostream>
#include<assert.h>
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

//SHARED 2D CONVOLUTION
//Also uses constant memory
//This is done assuming we know the tile width/ block dimensions
#define convolution1BlockDim 32
__global__ void sharedConvolution(unsigned char input[], unsigned char output[], int rows, int cols, int maskWidth){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    //Assuming we know the blocm width, to use 2D notation
    __shared__ unsigned char sharedM[convolution1BlockDim][convolution1BlockDim];
    //unsigned char pixVal = 0;

    int outputSizeSquare = 28; //We will produce 28 pixel values for each 32 threads (per dimension)

    //Bring in the data
    int colIndexShared = blockIdx.x * outputSizeSquare - maskWidth / 2 + threadIdx.x;
    int rowIndexShared = blockIdx.y * outputSizeSquare - maskWidth / 2 + threadIdx.y;
    
    if(colIndexShared > -1 && colIndexShared < cols && rowIndexShared > -1 && rowIndexShared < rows){
        sharedM[threadIdx.y][threadIdx.x] = input[rowIndexShared * cols + colIndexShared];
    }
    else {
        sharedM[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();
    
    if(row < rows && col < cols){
        if(threadIdx.x < outputSizeSquare && threadIdx.y < outputSizeSquare){
            unsigned char pixVal = 0;
            for(int j = 0; j < maskWidth; j++){
                for(int k = 0; k < maskWidth; k++){
                    pixVal += sharedM[threadIdx.y + j][threadIdx.x + k] * consMask[j * maskWidth + k];
                }
            }
            int colOutputIndex = blockIdx.x * outputSizeSquare + threadIdx.x;
            int rowOutputIndex = blockIdx.y * outputSizeSquare + threadIdx.y;
            output[rowOutputIndex * cols + colOutputIndex] = pixVal;
        }
    }
    
    // if(row < rows && col < cols){
    //     int startCol = col - maskWidth / 2;
    //     int startRow = row - maskWidth / 2;

    //     for(int j = 0; j < maskWidth; j++){
    //         for(int k = 0; k < maskWidth; k++){
    //             int curRow = startRow + j;
    //             int curCol = startCol + k;

    //             if(curRow > -1 && curRow < rows && curCol > -1 && curCol < cols){
    //                 pixVal += input[curRow * cols + curCol] * consMask[j * maskWidth + k];
    //             }
    //         }
    //     }
    // }

    //output[row * cols + col] = pixVal;
}

//SHARED 2D CONVOLUTION NO DIVERGENCE
//Also uses constant memory
//This is done assuming we know the tile width/ block dimensions
__global__ void sharedConvolutionDivergence(unsigned char input[], unsigned char output[], int rows, int cols, int maskWidth){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    //Assuming we know the blocm width, to use 2D notation
    __shared__ unsigned char sharedM[convolution1BlockDim][convolution1BlockDim];
    //unsigned char pixVal = 0;

    int outputSizeSquare = 28; //We will produce 28 pixel values for each 32 threads (per dimension)

    //Bring in the data
    int colIndexShared = blockIdx.x * outputSizeSquare - maskWidth / 2 + threadIdx.x;
    int rowIndexShared = blockIdx.y * outputSizeSquare - maskWidth / 2 + threadIdx.y;
    
    if(colIndexShared > -1 && colIndexShared < cols && rowIndexShared > -1 && rowIndexShared < rows){
        sharedM[threadIdx.y][threadIdx.x] = input[rowIndexShared * cols + colIndexShared];
    }
    else {
        sharedM[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();
    
    if(row < rows && col < cols){
        int indexVal = threadIdx.x + threadIdx.y * blockDim.x;
        if(indexVal < (blockDim.x * blockDim.y - (maskWidth - 1) * blockDim.y)){
            unsigned char pixVal = 0;
            int newIndex1D = ((indexVal / outputSizeSquare) * (maskWidth - 1) + indexVal);
            int newIndex1Dy = newIndex1D / blockDim.x;
            int newIndex1Dx = newIndex1D % blockDim.x;
            for(int j = 0; j < maskWidth; j++){
                for(int k = 0; k < maskWidth; k++){
                    pixVal += sharedM[newIndex1Dy + j][newIndex1Dx + k] * consMask[j * maskWidth + k];
                }
            }
            //int colOutputIndex = blockIdx.x * outputSizeSquare + threadIdx.x;
            //int rowOutputIndex = blockIdx.y * outputSizeSquare + threadIdx.y;
            int colOutputIndex = blockIdx.x * outputSizeSquare + newIndex1Dx;
            int rowOutputIndex = blockIdx.y * outputSizeSquare + newIndex1Dy;
            output[rowOutputIndex * cols + colOutputIndex] = pixVal;
        }
    }
    
}

#define CHANNELS 3 // we have 3 channels corresponding to RGB
// The input image is encoded as unsigned characters [0, 255]
__global__ void colorConvert(unsigned char * rgbImage,
    unsigned char * grayImage,
    int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {
        // get 1D coordinate for the grayscale image
        int grayOffset = y*width + x;
        // one can think of the RGB image having
        // CHANNEL times columns than the gray scale image
        int rgbOffset = grayOffset*CHANNELS;
        unsigned char b = rgbImage[rgbOffset ]; // red value for pixel
        unsigned char g = rgbImage[rgbOffset + 1]; // green value for pixel
        unsigned char r = rgbImage[rgbOffset + 2]; // blue value for pixel
        // perform the rescaling and store it
        // We multiply by floating point constants
        grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}



//GPU Convolution Validation
void gpuConvolutionTest(void){

//--------------------SERIAL STARTUP FIRST--------------------
    srand(time(NULL)); //Set up random values
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //We will compute a convolution image on a 720p made up image
    unsigned char inputImage[720][1280];
    unsigned char outputImage[720][1280];
    unsigned char mask5x5[5][5];

    //Initialize input image 
    for(int i = 0; i < 720; i++){
        for(int j = 0; j < 1280; j++){
            inputImage[i][j] = abs(rand() % 10);
        }
    }

    //Initialize input mask
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

    cudaEventRecord(start);
    
    for(int i=0; i < 720; ++i)              // rows
    {
        for(int j=0; j < 1280; ++j)          // columns
        {
            int startRow = i - kRowDisplace; //Starting Row
            int startCol = j - kColDisplace; //Starting Column
            unsigned char sum = 0; 
            
            for(int m=0; m < kRows; ++m) { //Kernel rows
                for(int n=0; n < kCols; ++n) { //Kernel Cols
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

//--------------------CUDA NAIVE IMPLEMENTATION--------------------

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
    //Events
    cudaEvent_t startMemoryTransfer, stopMemoryTransfer;
    cudaEventCreate(&startMemoryTransfer);
    cudaEventCreate(&stopMemoryTransfer);
    cudaEventRecord(startMemoryTransfer);

    err = cudaMemcpy(deviceInput, hostInput,
                            rows * cols * sizeof(unsigned char),
                            cudaMemcpyHostToDevice);
                            cudaEventRecord(stopNaive);
    
    cudaEventRecord(stopMemoryTransfer);
    cudaEventSynchronize(stopMemoryTransfer);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startMemoryTransfer, stopMemoryTransfer);
    std::cout << "Elapsed time for image transfer: " << milliseconds <<std::endl;

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
    
//--------------------CONSTANT MEMORY--------------------

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
    
//--------------------SHARED MEMORY--------------------

    //CUDA SHARED
    cudaEvent_t startShared, stopShared;
    cudaEventCreate(&startShared);
    cudaEventCreate(&stopShared);

    cudaEventRecord(startShared);
    int gridTileSize = 28;
    //Readjust grid
    dim3 sharedGridDim( 1 + (cols - 1) / gridTileSize,
                1 + (rows - 1) / gridTileSize);
    sharedConvolution<<<sharedGridDim, blockDim>>>
        (deviceInput, deviceOutput, rows, cols, maskVal);
    cudaEventRecord(stopShared);
    cudaEventSynchronize(stopShared);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startShared, stopShared);
    std::cout << "Elapsed time for shared: " << milliseconds <<std::endl;


    //Re-checking
    cudaEvent_t startMemoryReturn, stopMemoryReturn;
    cudaEventCreate(&startMemoryReturn);
    cudaEventCreate(&stopMemoryReturn);
    cudaEventRecord(startMemoryReturn);
    err = cudaMemcpy(hostOutput, deviceOutput,
                            rows * cols * sizeof(unsigned char),
                            cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stopMemoryReturn);
    cudaEventSynchronize(stopMemoryReturn);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startMemoryReturn, stopMemoryReturn);
    std::cout << "Elapsed time for image to return to CPU: " << milliseconds <<std::endl;
    cudaDeviceSynchronize();

    //Check wether output is correct
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            assert(hostOutput[i * cols + j] == outputImage[i][j]);
        }
    }


//--------------------SHARED MEMORY LESS DIVERGENCE--------------------
//--------------------THIS PROTOTYPE DID NOT WORK---------------------
    //CUDA LESS DIVERGENCE
    //cudaEvent_t startShared, stopShared;
    //cudaEventCreate(&startShared);
    //cudaEventCreate(&stopShared);

    //cudaEventRecord(startShared);
    //int gridTileSize = 28;
    //Readjust grid
    //dim3 sharedGridDim( 1 + (cols - 1) / gridTileSize,
    //            1 + (rows - 1) / gridTileSize);
    // sharedConvolutionDivergence<<<sharedGridDim, blockDim>>>
    //     (deviceInput, deviceOutput, rows, cols, maskVal);
    // cudaEventRecord(stopShared);
    // cudaEventSynchronize(stopShared);
    // milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, startShared, stopShared);
    // std::cout << "Elapsed time for divergence: " << milliseconds <<std::endl;

    //Memory Freeing
    //CUDA
    cudaFree(deviceInput);
    cudaFree(deviceMask);
    cudaFree(deviceOutput);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(startMemoryTransfer);
    cudaEventDestroy(stopMemoryTransfer);
    cudaEventDestroy(startNaive);
    cudaEventDestroy(stopNaive);
    cudaEventDestroy(startConstant);
    cudaEventDestroy(stopConstant);
    cudaEventDestroy(startShared);
    cudaEventDestroy(stopShared);
    //cudaEventDestroy(startDivergence);
    //cudaEventDestroy(stopDivergence);
    

    //CPU
    free(hostInput);
    free(hostMask);
    free(hostOutput);
}

__constant__ unsigned char consMask1[maskDimension*maskDimension];
__constant__ unsigned char consMask2[maskDimension*maskDimension];
__constant__ unsigned char consMask3[maskDimension*maskDimension];

void sampleKernelSingle(){
//----------TEST IMAGE GENERATION----------
    //Create a 720x1280 image
    //ASSUMED format is 3 channels BGR
    //tested with 5x5 masks
    int rows = 720;
    int columns = 1280;
    float milliseconds = 0;
    //Channel is defined as 3 above
    srand(time(NULL)); //Set up random values

    unsigned char inputImage[720][1280][CHANNELS];
    unsigned char* hostMask;
    unsigned char mask1[5][5];
    unsigned char mask2[5][5];
    unsigned char mask3[5][5];

    //Initialize Input Image
    for(int i = 0; i < 720; i++){
        for(int j = 0; j < 1280; j++){
            for(int k = 0; k < CHANNELS; k++){
                inputImage[i][j][k] = (char)rand();
            }
        }
    }

    //Initialize all three masks
    for(int i = 0; i < 1280; i++){
        for(int j = 0; j < CHANNELS; j++){
            mask1[i][j] = (char)rand();
        }
    }

    //Initialize all three masks
    for(int i = 0; i < 1280; i++){
        for(int j = 0; j < CHANNELS; j++){
            mask2[i][j] = (char)rand();
        }
    }

    //Initialize all three masks
    for(int i = 0; i < 1280; i++){
        for(int j = 0; j < CHANNELS; j++){
            mask3[i][j] = (char)rand();
        }
    }


//----------CUDA BEGIN----------
    //Allocate memory for all arrays and intermediate results
    //Memory Declarations and Allocations
    unsigned char* hostBGRInput;
    unsigned char* deviceBGRInput;
    unsigned char* hostGrayImage;
    unsigned char* deviceGrayImage;
    unsigned char* hostConvolved1;
    unsigned char* deviceConvolved1;
    unsigned char* hostConvolved2;
    unsigned char* deviceConvolved2;
    unsigned char* hostConvolved3;
    unsigned char* deviceConvolved3;

    //Malloc Host
    hostBGRInput = (unsigned char*)malloc(rows * columns * CHANNELS * sizeof(unsigned char));
    hostGrayImage = (unsigned char*)malloc(rows * columns * sizeof(unsigned char));
    hostConvolved1 = (unsigned char*)malloc(rows * columns * sizeof(unsigned char));
    hostConvolved2 = (unsigned char*)malloc(rows * columns * sizeof(unsigned char));
    hostConvolved3 = (unsigned char*)malloc(rows * columns  * sizeof(unsigned char));

    //CudaMalloc
    err = cudaMalloc((void**)&deviceBGRInput, rows * columns * CHANNELS * sizeof(unsigned char));
    err = cudaMalloc((void**)&deviceGrayImage, rows * columns * sizeof(unsigned char));
    err = cudaMalloc((void**)&deviceConvolved1, rows * columns * sizeof(unsigned char));
    err = cudaMalloc((void**)&deviceConvolved2, rows * columns * sizeof(unsigned char));
    err = cudaMalloc((void**)&deviceConvolved3, rows * columns * sizeof(unsigned char));

    //Generate input array on Host
    //Initialize Input Image
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            for(int k = 0; k < CHANNELS; k++){
                hostBGRInput[i * columns * CHANNELS + j * CHANNELS + k] = inputImage[i][j][k];
            }
        }
    }

    //Dimensions for blocks and grayscale grid
    int blockSize = 32;
    int maskVal = 5;
    int gridTileSize = 28;
    dim3 blockDim(blockSize,blockSize), gridDim( 1 + (columns - 1) / blockSize,
                                             1 + (rows - 1) / blockSize);
    //Grid for convolution shared
    dim3 sharedGridDim( 1 + (columns - 1) / gridTileSize,
                1 + (rows - 1) / gridTileSize);

    //Constant memory
    hostMask = (unsigned char*)malloc(maskVal * maskVal * sizeof(unsigned char));
    for(int i = 0; i < maskVal; i++){
        for(int j = 0; j < maskVal; j++){
            hostMask[i * maskVal + j] = mask1[i][j];
        }
    }
    cudaMemcpyToSymbol(consMask, hostMask, maskVal * maskVal * sizeof(unsigned char));

    //BEGIN OPERATIONS
    //Copy Input array to GPU
    cudaEvent_t kernelLaunch, kernelEnd;
    cudaEventCreate(&kernelLaunch);
    cudaEventCreate(&kernelEnd);
    cudaEventRecord(kernelLaunch);

    err = cudaMemcpy(deviceBGRInput, hostBGRInput, rows * columns * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    colorConvert<<<gridDim, blockDim>>>(deviceBGRInput, deviceGrayImage, columns, rows);

    sharedConvolution<<<sharedGridDim, blockDim>>>
        (deviceGrayImage, deviceConvolved1, rows, columns, maskVal);
    sharedConvolution<<<sharedGridDim, blockDim>>>
        (deviceConvolved1, deviceConvolved2, rows, columns, maskVal);
    sharedConvolution<<<sharedGridDim, blockDim>>>
        (deviceConvolved2, deviceConvolved3, rows, columns, maskVal);

    err = cudaMemcpy(hostConvolved3, deviceConvolved3, rows * columns * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaEventRecord(kernelEnd);
    cudaEventSynchronize(kernelEnd);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, kernelLaunch, kernelEnd);
    printf("Total Single Kernel time: %fms\n", milliseconds);
    printf("estimated time for 30 images: %fms\n", milliseconds * 30);





    //FREE HOST
    free(hostBGRInput);
    free(hostGrayImage);
    free(hostConvolved1);
    free(hostConvolved2);
    free(hostConvolved3);

    //FREE DEVICE
    cudaFree(deviceBGRInput);
    cudaFree(deviceGrayImage);
    cudaFree(deviceConvolved1);
    cudaFree(deviceConvolved2);
    cudaFree(deviceConvolved3);
}

//--------------------------------STREAMING TEST-----------------------------------------
#define stream_size 2
#define num_images 10
void betaStreamTest(){
    //----------TEST IMAGE GENERATION----------
    //Create a 720x1280 image
    //ASSUMED format is 3 channels BGR
    //tested with 5x5 masks
    int rows = 720;
    int columns = 1280;
    float milliseconds = 0;
    //Channel is defined as 3 above
    srand(time(NULL)); //Set up random values

    unsigned char inputImage[720][1280][CHANNELS];
    unsigned char* hostMask;
    unsigned char mask1[5][5];
    unsigned char mask2[5][5];
    unsigned char mask3[5][5];

    //Initialize Input Image
    for(int i = 0; i < 720; i++){
        for(int j = 0; j < 1280; j++){
            for(int k = 0; k < CHANNELS; k++){
                inputImage[i][j][k] = (char)rand();
            }
        }
    }

    //Initialize all three masks
    for(int i = 0; i < 1280; i++){
        for(int j = 0; j < CHANNELS; j++){
            mask1[i][j] = (char)rand();
        }
    }

    //Initialize all three masks
    for(int i = 0; i < 1280; i++){
        for(int j = 0; j < CHANNELS; j++){
            mask2[i][j] = (char)rand();
        }
    }

    //Initialize all three masks
    for(int i = 0; i < 1280; i++){
        for(int j = 0; j < CHANNELS; j++){
            mask3[i][j] = (char)rand();
        }
    }


//----------CUDA BEGIN----------
    //Allocate memory for all arrays and intermediate results
    //Memory Declarations and Allocations
    unsigned char* hostBGRInput[num_images];
    unsigned char* deviceBGRInput[num_images];
    unsigned char* hostGrayImage[num_images];
    unsigned char* deviceGrayImage[num_images];
    unsigned char* hostConvolved1[num_images];
    unsigned char* deviceConvolved1[num_images];
    unsigned char* hostConvolved2[num_images];
    unsigned char* deviceConvolved2[num_images];
    unsigned char* hostConvolved3[num_images];
    unsigned char* deviceConvolved3[num_images];

    //Malloc Host
    for (int i = 0; i < num_images; i++){
        cudaHostAlloc( (void**)&hostBGRInput[i], rows * columns * CHANNELS * sizeof(unsigned char), cudaHostAllocDefault);
        cudaHostAlloc( (void**)&hostGrayImage[i], rows * columns * sizeof(unsigned char), cudaHostAllocDefault);
        cudaHostAlloc( (void**)&hostConvolved1[i], rows * columns * sizeof(unsigned char), cudaHostAllocDefault);
        cudaHostAlloc( (void**)&hostConvolved2[i], rows * columns * sizeof(unsigned char), cudaHostAllocDefault);
        cudaHostAlloc( (void**)&hostConvolved3[i], rows * columns * sizeof(unsigned char), cudaHostAllocDefault);

        // hostBGRInput[i] = (unsigned char*)malloc(rows * columns * CHANNELS * sizeof(unsigned char));
        // hostGrayImage[i] = (unsigned char*)malloc(rows * columns * sizeof(unsigned char));
        // hostConvolved1[i] = (unsigned char*)malloc(rows * columns * sizeof(unsigned char));
        // hostConvolved2[i] = (unsigned char*)malloc(rows * columns * sizeof(unsigned char));
        // hostConvolved3[i] = (unsigned char*)malloc(rows * columns  * sizeof(unsigned char));
    }

    //CudaMalloc
    for(int i = 0; i < num_images; i++){
        err = cudaMalloc((void**)&deviceBGRInput[i], rows * columns * CHANNELS * sizeof(unsigned char));
        err = cudaMalloc((void**)&deviceGrayImage[i], rows * columns * sizeof(unsigned char));
        err = cudaMalloc((void**)&deviceConvolved1[i], rows * columns * sizeof(unsigned char));
        err = cudaMalloc((void**)&deviceConvolved2[i], rows * columns * sizeof(unsigned char));
        err = cudaMalloc((void**)&deviceConvolved3[i], rows * columns * sizeof(unsigned char));
    }

    //Generate input array on Host
    //Initialize Input Image
    for(int s = 0; s < num_images; s++){
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < columns; j++){
                for(int k = 0; k < CHANNELS; k++){
                    hostBGRInput[s][i * columns * CHANNELS + j * CHANNELS + k] = inputImage[i][j][k];
                }
            }
        }
    }
    //Dimensions for blocks and grayscale grid
    int blockSize = 32;
    int maskVal = 5;
    int gridTileSize = 28;
    dim3 blockDim(blockSize,blockSize), gridDim( 1 + (columns - 1) / blockSize,
                                             1 + (rows - 1) / blockSize);
    //Grid for convolution shared
    dim3 sharedGridDim( 1 + (columns - 1) / gridTileSize,
                1 + (rows - 1) / gridTileSize);

    //Constant memory
    hostMask = (unsigned char*)malloc(maskVal * maskVal * sizeof(unsigned char));
    for(int i = 0; i < maskVal; i++){
        for(int j = 0; j < maskVal; j++){
            hostMask[i * maskVal + j] = mask1[i][j];
        }
    }
    cudaMemcpyToSymbol(consMask, hostMask, maskVal * maskVal);

    //Initialize stream
    cudaStream_t streams[stream_size];
    for(int i = 0; i < stream_size; i++){
        cudaStreamCreate(&streams[i]);
    }

    //BEGIN OPERATIONS
    //Copy Input array to GPU
    cudaEvent_t kernelLaunch, kernelEnd;
    cudaEventCreate(&kernelLaunch);
    cudaEventCreate(&kernelEnd);
    cudaEventRecord(kernelLaunch);

    for(int j = 0; j < num_images; j = j + stream_size){
        for(int i = 0; i < stream_size; i++){
            err = cudaMemcpyAsync(deviceBGRInput[i + j], hostBGRInput[i + j], rows * columns * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice, streams[i]);
            
            colorConvert<<<gridDim, blockDim, 0, streams[i]>>>(deviceBGRInput[i + j], deviceGrayImage[i + j], columns, rows);

            sharedConvolution<<<sharedGridDim, blockDim, 0, streams[i]>>>
                (deviceGrayImage[i + j], deviceConvolved1[i + j], rows, columns, maskVal);
            sharedConvolution<<<sharedGridDim, blockDim, 0, streams[i]>>>
                (deviceConvolved1[i + j], deviceConvolved2[i + j], rows, columns, maskVal);
            sharedConvolution<<<sharedGridDim, blockDim, 0, streams[i]>>>
                (deviceConvolved2[i + j], deviceConvolved3[i + j], rows, columns, maskVal);   

            err = cudaMemcpyAsync(hostConvolved3[i + j], deviceConvolved3[i + j], rows * columns * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[i]);
            cudaStreamSynchronize(streams[i]);
        }
    }
    cudaDeviceSynchronize();
    cudaEventRecord(kernelEnd);
    cudaEventSynchronize(kernelEnd);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, kernelLaunch, kernelEnd);
    printf("Total 30 Kernel time: %fms\n", milliseconds);
    printf("estimated time for 30 images: %fms\n", milliseconds * 1);





    //FREE HOST
    // free(hostBGRInput);
    // free(hostGrayImage);
    // free(hostConvolved1);
    // free(hostConvolved2);
    // free(hostConvolved3);

    //FREE DEVICE
    for(int i = 0; i < num_images; i++){
        cudaFreeHost(hostBGRInput[i]);
        cudaFreeHost(hostGrayImage[i]);
        cudaFreeHost(hostConvolved1[i]);
        cudaFreeHost(hostConvolved2[i]);
        cudaFreeHost(hostConvolved3[i]);
        cudaFree(deviceBGRInput[i]);
        cudaFree(deviceGrayImage[i]);
        cudaFree(deviceConvolved1[i]);
        cudaFree(deviceConvolved2[i]);
        cudaFree(deviceConvolved3[i]);
    }
    
}

// int main(void){
//     printf("Testing basic GPU convolution compatibility\n");
//     gpuConvolutionTest();
//     printf("\n\nTesting images with single stream element.\n");
//     sampleKernelSingle();
//     printf("\n\nTesting with 5 streams.\n");
//     betaStreamTest();
// }