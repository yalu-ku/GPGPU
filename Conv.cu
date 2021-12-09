#include <stdio.h>

void verify1D(float * N, float * P, float * mask, int width, int mask_width);

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define WIDTH 10000 //Input Vector 길이
#define MASK_WIDTH 5
#define O_TILE_WIDTH 1020
#define BLOCK_WIDTH (O_TILE_WIDTH + 4)

__global__ void Conv1D(float * N, float * P, float * Mask, int width, int mask_width)
{
    float output = 0.0f;
    int tx = threadIdx.x;
    int index_o = blockIdx.x * O_TILE_WIDTH + threadIdx.x;
    int index_i = index_o - 2;
    __shared__ float Ns[BLOCK_WIDTH];

    if((index_i>=0) && (index_i < width)) {
        Ns[tx] = N[index_i];
    } else {
        Ns[tx] = 0.0f;
    }
    __syncthreads();

    if((threadIdx.x < O_TILE_WIDTH) && (index_o < width)) {
        output = 0.0f;
        for(int j=0; j < mask_width; j++) {
            output += Mask[j] * Ns[j+threadIdx.x];
        }
        P[index_o] = output;
    }
}

int main()
{
    float *N, *P, *Mask;
    float *dev_N, *dev_P, *dev_Mask;

    N = (float*)malloc(sizeof(float)*WIDTH);
    P = (float*)malloc(sizeof(float)*WIDTH);
    Mask = (float*)malloc(sizeof(float)*5);

    // Initialize
    for (int i=0; i<WIDTH; i++) {
        N[i] = (rand()%100)/100.00;
    }
    for (int i=0; i<5; i++) {
        Mask[i] = (rand()%100)/100.00;
    }

    // Add vectors in parrallel
    HANDLE_ERROR(cudaMalloc((void**)&dev_N, WIDTH * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_P, WIDTH * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_Mask, WIDTH * sizeof(float)));

    HANDLE_ERROR(cudaMemcpy(dev_N, N, WIDTH * sizeof(float), cudaMemcpyHostToDevice));    HANDLE_ERROR(cudaMemcpy(dev_Mask, Mask, WIDTH * sizeof(float), cudaMemcpyHostToDevice));

    // Launch a kernel on the GPU with one thread for each element.
    dim3 dimBlock(BLOCK_WIDTH, 1, 1);
    dim3 dimGrid((WIDTH-1)/O_TILE_WIDTH+1, 1, 1);
    Conv1D<<<dimGrid, dimBlock>>>(dev_N, dev_P, dev_Mask, WIDTH, MASK_WIDTH);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    HANDLE_ERROR(cudaMemcpy(P, dev_P, WIDTH * sizeof(float), cudaMemcpyDeviceToHost));

    verify1D(N, P, Mask, WIDTH, MASK_WIDTH);

    cudaFree(dev_N);
    cudaFree(dev_P);
    cudaFree(dev_Mask);
    free(N);
    free(P);
    free(Mask);
    return 0;
}

void verify1D(float * N, float * P, float * mask, int width, int mask_width)
{
    const float relativeTolerance = 1e-6;
    for(int i=0; i<width; ++i) {
        float sum = 0.0f;
        for(int i_m=0; i_m<mask_width; ++i_m) {
            int iN = i + i_m - (int)(mask_width/2);
            if(iN>=0 && iN<width) {
                sum += mask[i_m]*N[iN];
            }
        }
        float relativeError = (sum - P[i])/sum;
        if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
            printf("TEST FAILED\n\n");
            exit(0);
        }
    }
    printf("TEST PASSED\n\n");
}