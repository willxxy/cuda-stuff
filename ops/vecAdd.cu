// vecAdd.cu

#include <cstdio>
#include <cuda_runtime.h>

// CUDA kernel: C[i] = A[i] + B[i]
__global__ void vecAddKernel(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int n = 1<<20;                   // total elements (e.g. 1 048 576)
    const size_t bytes = n * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize
    for (int i = 0; i < n; ++i) {
        h_A[i] = float(i);
        h_B[i] = float(2*i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;
    vecAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf("Mismatch at %d: %f != %f\n", i, h_C[i], h_A[i]+h_B[i]);
            ok = false;
            break;
        }
    }
    printf("Result %s\n", ok ? "correct" : "INCORRECT");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return ok ? 0 : 1;
}
