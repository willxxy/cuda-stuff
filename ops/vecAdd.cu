// vecAdd.cu

#include <cstdio>
#include <iostream> // Added for C++ style output
#include <cuda_runtime.h>

// Helper function to check for CUDA errors
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " (" << cudaGetErrorString(result) << ") at " <<
            file << ":" << line << " '" << func << "'" << std::endl;
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}


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
    std::cout << "Vector size (n): " << n << std::endl;
    std::cout << "Total bytes: " << bytes << std::endl;

    // Allocate host memory
    std::cout << "Allocating host memory..." << std::endl;
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (h_A == nullptr || h_B == nullptr || h_C == nullptr) {
        std::cerr << "Failed to allocate host memory!" << std::endl;
        return 1;
    }
    std::cout << "Host memory allocated." << std::endl;


    // Initialize
    std::cout << "Initializing host vectors..." << std::endl;
    for (int i = 0; i < n; ++i) {
        h_A[i] = float(i);
        h_B[i] = float(2*i);
    }
    std::cout << "Host vectors initialized." << std::endl;

    // Allocate device memory
    std::cout << "Allocating device memory..." << std::endl;
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc(&d_A, bytes));
    checkCudaErrors(cudaMalloc(&d_B, bytes));
    checkCudaErrors(cudaMalloc(&d_C, bytes));
    std::cout << "Device memory allocated." << std::endl;

    // Copy inputs to device
    std::cout << "Copying input data from host to device..." << std::endl;
    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    std::cout << "Input data copied to device." << std::endl;

    // Launch kernel
    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;
    std::cout << "Launching kernel with grid size " << gridSize << " and block size " << blockSize << "..." << std::endl;
    vecAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    checkCudaErrors(cudaGetLastError()); // Check for errors during kernel launch
    std::cout << "Kernel launched. Synchronizing device..." << std::endl;
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "Device synchronized." << std::endl;

    // Copy result back
    std::cout << "Copying result data from device to host..." << std::endl;
    checkCudaErrors(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    std::cout << "Result data copied to host." << std::endl;

    // Sample additions
    std::cout << "Sample additions:\n";
    for (int i = 0; i < 8; ++i) {
        std::cout
        << "  h_A[" << i << "] = " << h_A[i]
        << " + h_B[" << i << "] = " << h_B[i]
        << " → h_C[" << i << "] = " << h_C[i]
        << "\n";
    }

    // Verify
    std::cout << "Verifying result..." << std::endl;
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabsf(h_C[i] - expected) > 1e-5) { // Use tolerance for float comparison
            printf("Mismatch at %d: %f (actual) != %f (expected)\n", i, h_C[i], expected);
            ok = false;
            break;
        }
    }
    printf("Result %s\n", ok ? "correct" : "INCORRECT");

    // Cleanup
    std::cout << "Cleaning up device memory..." << std::endl;
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    std::cout << "Device memory freed." << std::endl;
    std::cout << "Cleaning up host memory..." << std::endl;
    free(h_A);
    free(h_B);
    free(h_C);
    std::cout << "Host memory freed." << std::endl;

    // Reset device at the end
    checkCudaErrors(cudaDeviceReset());

    std::cout << "Execution finished." << std::endl;
    return ok ? 0 : 1;
}
