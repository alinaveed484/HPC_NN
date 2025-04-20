#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cublas_v2.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f
#define EPOCHS 3
#define NUM_CLASSES 10

// Check CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Check cuBLAS errors
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d - %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate contiguous memory
float* allocateContiguous(int numElements) {
    float* data = (float*)malloc(numElements * sizeof(float));
    if (!data) {
        fprintf(stderr, "Error allocating memory\n");
        exit(EXIT_FAILURE);
    }
    return data;
}

// Neural Network structure
typedef struct {
    float* W1;  // INPUT_SIZE x HIDDEN_SIZE
    float* W2;  // HIDDEN_SIZE x OUTPUT_SIZE
    float* b1;  // HIDDEN_SIZE
    float* b2;  // OUTPUT_SIZE
} NeuralNetwork;

NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = (float*)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    net->W2 = (float*)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    net->b1 = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    net->b2 = (float*)calloc(OUTPUT_SIZE, sizeof(float));

    srand(777);
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        net->W1[i] = ((float)rand() / RAND_MAX) * 0.01f;
    }
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        net->W2[i] = ((float)rand() / RAND_MAX) * 0.01f;
    }
    return net;
}

void freeNetwork(NeuralNetwork* net) {
    free(net->W1);
    free(net->W2);
    free(net->b1);
    free(net->b2);
    free(net);
}

// Kernels
__global__ void add_bias_and_relu(float* hidden, float* b1, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float sum = hidden[i] + b1[i];
        hidden[i] = (sum > 0.0f) ? sum : 0.0f;
    }
}

__global__ void add_bias_and_softmax(float* output, float* b2, int size) {
    int i = threadIdx.x;
    if (i < size) {
        output[i] += b2[i];
    }
    __syncthreads();
    if (i == 0) {
        float max_val = output[0];
        for (int k = 1; k < size; k++) {
            if (output[k] > max_val) max_val = output[k];
        }
        float sum = 0.0f;
        for (int k = 0; k < size; k++) {
            output[k] = expf(output[k] - max_val);
            sum += output[k];
        }
        if (sum == 0.0f) {
            sum = 1e-8f;  // Prevent division by zero
        }
        for (int k = 0; k < size; k++) {
            output[k] /= sum;
        }
    }
}

__global__ void compute_d_output(float* output, float* target, float* d_output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < OUTPUT_SIZE) {
        d_output[i] = output[i] - target[i];
    }
}

__global__ void apply_relu_derivative(float* d_hidden, float* hidden, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_hidden[i] *= (hidden[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

__global__ void update_W2(float* W2, float* d_output, float* hidden, float lr) {
    int j = threadIdx.x;  // HIDDEN_SIZE
    int i = blockIdx.x;   // OUTPUT_SIZE
    if (j < HIDDEN_SIZE && i < OUTPUT_SIZE) {
        W2[j * OUTPUT_SIZE + i] -= lr * d_output[i] * hidden[j];
    }
}

__global__ void update_W1(float* W1, float* d_hidden, float* input, float lr) {
    int i = blockIdx.x;    // INPUT_SIZE
    int j = threadIdx.x;   // HIDDEN_SIZE
    if (i < INPUT_SIZE && j < HIDDEN_SIZE) {
        int idx = i * HIDDEN_SIZE + j;
        W1[idx] -= lr * d_hidden[j] * input[i];
    }
}

__global__ void update_b1(float* b1, float* d_hidden, float lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN_SIZE) {
        b1[i] -= lr * d_hidden[i];
    }
}

__global__ void update_b2(float* b2, float* d_output, float lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < OUTPUT_SIZE) {
        b2[i] -= lr * d_output[i];
    }
}

// Data loading
float* loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fseek(file, 16, SEEK_SET);
    float* images = allocateContiguous(numImages * INPUT_SIZE);
    for (int i = 0; i < numImages * INPUT_SIZE; i++) {
        unsigned char pixel;
        fread(&pixel, sizeof(unsigned char), 1, file);
        images[i] = pixel / 255.0f;
    }
    fclose(file);
    return images;
}

float* loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fseek(file, 8, SEEK_SET);
    float* labels = allocateContiguous(numLabels * OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, file);
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i * OUTPUT_SIZE + j] = (j == label) ? 1.0f : 0.0f;
        }
    }
    fclose(file);
    return labels;
}

// Training function
void train(NeuralNetwork* net, float* d_W1, float* d_W2, float* d_b1, float* d_b2,
           float* H_images, float* H_labels, int numImages, cublasHandle_t handle) {
    float *D_images, *D_labels;
    CUDA_CHECK(cudaMalloc(&D_images, numImages * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D_labels, numImages * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(D_images, H_images, numImages * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D_labels, H_labels, numImages * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    float *D_hidden, *D_output, *D_d_output, *D_d_hidden;
    CUDA_CHECK(cudaMalloc(&D_hidden, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D_output, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D_d_output, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D_d_hidden, HIDDEN_SIZE * sizeof(float)));

    cudaEvent_t f_start, f_stop, b_start, b_stop;
    CUDA_CHECK(cudaEventCreate(&f_start));
    CUDA_CHECK(cudaEventCreate(&f_stop));
    CUDA_CHECK(cudaEventCreate(&b_start));
    CUDA_CHECK(cudaEventCreate(&b_stop));

    cudaStream_t compute_stream, copy_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&copy_stream));

    cudaEvent_t copy_event;
    CUDA_CHECK(cudaEventCreate(&copy_event));

    float* h_output;
    CUDA_CHECK(cudaMallocHost((void**)&h_output, OUTPUT_SIZE * sizeof(float)));

    float alpha = 1.0f, beta = 0.0f;

    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        float loss = 0.0f, forward_ms = 0.0f, backward_ms = 0.0f;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            float* d_input = D_images + i * INPUT_SIZE;
            float* d_label = D_labels + i * OUTPUT_SIZE;

            // Forward Pass
            CUDA_CHECK(cudaEventRecord(f_start, compute_stream));
            // hidden = W1 * input (W1 is INPUT_SIZE x HIDDEN_SIZE row-major)
            CUBLAS_CHECK(cublasSetStream(handle, compute_stream));
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     HIDDEN_SIZE, 1, INPUT_SIZE,
                                     &alpha, d_W1, HIDDEN_SIZE, d_input, INPUT_SIZE,
                                     &beta, D_hidden, HIDDEN_SIZE));
            add_bias_and_relu<<<(HIDDEN_SIZE + 255)/256, 256, 0, compute_stream>>>(D_hidden, d_b1, HIDDEN_SIZE);
            CUDA_CHECK(cudaGetLastError());
            // output = W2 * hidden (W2 is HIDDEN_SIZE x OUTPUT_SIZE row-major)
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     OUTPUT_SIZE, 1, HIDDEN_SIZE,
                                     &alpha, d_W2, OUTPUT_SIZE, D_hidden, HIDDEN_SIZE,
                                     &beta, D_output, OUTPUT_SIZE));
            add_bias_and_softmax<<<1, OUTPUT_SIZE, 0, compute_stream>>>(D_output, d_b2, OUTPUT_SIZE);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(f_stop, compute_stream));
            CUDA_CHECK(cudaEventSynchronize(f_stop));
            {
                float ms;
                cudaEventElapsedTime(&ms, f_start, f_stop);
                forward_ms += ms;
            }

            // Backward Pass
            CUDA_CHECK(cudaEventRecord(b_start, compute_stream));
            compute_d_output<<<1, OUTPUT_SIZE, 0, compute_stream>>>(D_output, d_label, D_d_output);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(copy_event, compute_stream));
            CUDA_CHECK(cudaStreamWaitEvent(copy_stream, copy_event, 0));
            CUDA_CHECK(cudaMemcpyAsync(h_output, D_output, OUTPUT_SIZE * sizeof(float),
                                       cudaMemcpyDeviceToHost, copy_stream));
            // d_hidden = W2^T * d_output (W2 is HIDDEN_SIZE x OUTPUT_SIZE row-major)
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     HIDDEN_SIZE, 1, OUTPUT_SIZE,
                                     &alpha, d_W2, OUTPUT_SIZE, D_d_output, OUTPUT_SIZE,
                                     &beta, D_d_hidden, HIDDEN_SIZE));
            apply_relu_derivative<<<(HIDDEN_SIZE + 255)/256, 256, 0, compute_stream>>>(D_d_hidden, D_hidden, HIDDEN_SIZE);
            CUDA_CHECK(cudaGetLastError());
            update_W2<<<OUTPUT_SIZE, HIDDEN_SIZE, 0, compute_stream>>>(d_W2, D_d_output, D_hidden, LEARNING_RATE);
            CUDA_CHECK(cudaGetLastError());
            update_W1<<<INPUT_SIZE, HIDDEN_SIZE, 0, compute_stream>>>(d_W1, D_d_hidden, d_input, LEARNING_RATE);
            CUDA_CHECK(cudaGetLastError());
            update_b1<<<1, HIDDEN_SIZE, 0, compute_stream>>>(d_b1, D_d_hidden, LEARNING_RATE);
            CUDA_CHECK(cudaGetLastError());
            update_b2<<<1, OUTPUT_SIZE, 0, compute_stream>>>(d_b2, D_d_output, LEARNING_RATE);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(b_stop, compute_stream));
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));
            CUDA_CHECK(cudaStreamSynchronize(copy_stream));
            {
                float ms;
                cudaEventElapsedTime(&ms, b_start, b_stop);
                backward_ms += ms;
            }

            // Loss and accuracy
            CUDA_CHECK(cudaStreamSynchronize(copy_stream));
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                loss -= H_labels[i * OUTPUT_SIZE + k] * logf(h_output[k] + 1e-8f);
            }
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (h_output[j] > h_output[pred]) pred = j;
                if (H_labels[i * OUTPUT_SIZE + j] > H_labels[i * OUTPUT_SIZE + actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Acc: %.2f%% - Time: %.3fs - (FWD: %.3fs, BWD: %.3fs)\n",
               epoch + 1, loss / numImages, (correct * 100.0f) / numImages,
               get_time(epoch_start), forward_ms / 1000.0f, backward_ms / 1000.0f);
    }
    printf("Total Time: %.3fs\n", get_time(total_start));

    CUDA_CHECK(cudaFree(D_images));
    CUDA_CHECK(cudaFree(D_labels));
    CUDA_CHECK(cudaFree(D_hidden));
    CUDA_CHECK(cudaFree(D_output));
    CUDA_CHECK(cudaFree(D_d_output));
    CUDA_CHECK(cudaFree(D_d_hidden));
    CUDA_CHECK(cudaFreeHost(h_output));
    CUDA_CHECK(cudaEventDestroy(copy_event));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    CUDA_CHECK(cudaEventDestroy(f_start));
    CUDA_CHECK(cudaEventDestroy(f_stop));
    CUDA_CHECK(cudaEventDestroy(b_start));
    CUDA_CHECK(cudaEventDestroy(b_stop));
}
void evaluate(float* images, float* labels, int numImages,
              float* d_W1, float* d_W2, float* d_b1, float* d_b2,
              cublasHandle_t handle) {
    float *D_images = NULL, *D_hidden = NULL, *D_output = NULL;
    float alpha = 1.0f, beta = 0.0f;
    int correct = 0;
    float h_output[OUTPUT_SIZE];

    // Copy all test images to the device once
    CUDA_CHECK(cudaMalloc(&D_images, numImages * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(D_images,
                          images,
                          numImages * INPUT_SIZE * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Allocate space for one hidden‐layer activation and one output activation
    CUDA_CHECK(cudaMalloc(&D_hidden, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D_output, OUTPUT_SIZE * sizeof(float)));

    // Ensure cuBLAS ops run on the default stream
    CUBLAS_CHECK(cublasSetStream(handle, 0));

    for (int i = 0; i < numImages; ++i) {
        // Point to the i-th image on the device
        float* d_input = D_images + i * INPUT_SIZE;

        // --- Forward pass: input → hidden ---
        CUBLAS_CHECK(
            cublasSgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        HIDDEN_SIZE, 1, INPUT_SIZE,
                        &alpha,
                        d_W1, HIDDEN_SIZE,
                        d_input, INPUT_SIZE,
                        &beta,
                        D_hidden, HIDDEN_SIZE)
        );
        add_bias_and_relu<<<(HIDDEN_SIZE + 255) / 256, 256>>>(D_hidden, d_b1, HIDDEN_SIZE);
        CUDA_CHECK(cudaGetLastError());

        // --- Forward pass: hidden → output ---
        CUBLAS_CHECK(
            cublasSgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        OUTPUT_SIZE, 1, HIDDEN_SIZE,
                        &alpha,
                        d_W2, OUTPUT_SIZE,
                        D_hidden, HIDDEN_SIZE,
                        &beta,
                        D_output, OUTPUT_SIZE)
        );
        add_bias_and_softmax<<<1, OUTPUT_SIZE>>>(D_output, d_b2, OUTPUT_SIZE);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy the output probabilities back to host
        CUDA_CHECK(cudaMemcpy(h_output,
                              D_output,
                              OUTPUT_SIZE * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Compute prediction vs. ground truth
        int pred = 0, actual = 0;
        for (int j = 1; j < OUTPUT_SIZE; ++j) {
            if (h_output[j] > h_output[pred])       pred = j;
            if (labels[i * OUTPUT_SIZE + j] >
                labels[i * OUTPUT_SIZE + actual])   actual = j;
        }
        if (pred == actual) ++correct;
    }

    printf("Test Accuracy: %.2f%%\n",
           (correct * 100.0f) / (float)numImages);

    // Clean up
    CUDA_CHECK(cudaFree(D_images));
    CUDA_CHECK(cudaFree(D_hidden));
    CUDA_CHECK(cudaFree(D_output));
}

int main() {
    float* H_train_images = loadMNISTImages("../../data/train-images.idx3-ubyte", 60000);
    float* H_train_labels = loadMNISTLabels("../../data/train-labels.idx1-ubyte", 60000);
    float* H_test_images = loadMNISTImages("../../data/t10k-images.idx3-ubyte", 10000);
    float* H_test_labels = loadMNISTLabels("../../data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();

    float *d_W1, *d_W2, *d_b1, *d_b2;
    CUDA_CHECK(cudaMalloc(&d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, OUTPUT_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_W1, net->W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, net->W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, net->b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, net->b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    train(net, d_W1, d_W2, d_b1, d_b2, H_train_images, H_train_labels, 60000, handle);
    evaluate(H_test_images, H_test_labels, 10000, d_W1, d_W2, d_b1, d_b2, handle);

    CUBLAS_CHECK(cublasDestroy(handle));
    freeNetwork(net);
    free(H_train_images);
    free(H_train_labels);
    free(H_test_images);
    free(H_test_labels);
    CUDA_CHECK(cudaFree(d_W1));
    CUDA_CHECK(cudaFree(d_W2));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_b2));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}