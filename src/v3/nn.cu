#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define NUM_CLASSES 10

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate contiguous memory
double* allocateContiguous(int numElements) {
    double* data = (double*)malloc(numElements * sizeof(double));
    if (!data) {
        fprintf(stderr, "Error allocating memory\n");
        exit(EXIT_FAILURE);
    }
    return data;
}

// Neural Network structure
typedef struct {
    double* W1;  // HIDDEN_SIZE x INPUT_SIZE
    double* W2;  // OUTPUT_SIZE x HIDDEN_SIZE
    double* b1;  // HIDDEN_SIZE
    double* b2;  // OUTPUT_SIZE
} NeuralNetwork;

NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    net->W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(777);
    for (int j = 0; j < INPUT_SIZE; j++) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            net->W1[j * HIDDEN_SIZE + i] = ((double)rand() / RAND_MAX) * 0.01;
        }
    }

    // W2 initialization (HIDDEN_SIZE x OUTPUT_SIZE)
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            net->W2[j * OUTPUT_SIZE + i] = ((double)rand() / RAND_MAX) * 0.01;
        }
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

// Device Softmax
__device__ void kernelsoftmax(double* x, int size) {
    double max_x = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max_x) max_x = x[i];
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_x);
        sum += x[i];
    }
    if (sum == 0.0) {
        for (int i = 0; i < size; i++)
            x[i] = 1.0 / size;
    } else {
        for (int i = 0; i < size; i++)
            x[i] /= sum;
    }
}

// Forward Kernels
__global__ void compute_hidden(double* W1, double* b1, double* input, double* hidden) {

    extern __shared__ double s_input[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Load input into shared memory
    for (int j = tid; j < INPUT_SIZE; j += blockDim.x) {
        s_input[j] = input[j];
    }
    __syncthreads();

    if (i < HIDDEN_SIZE) {
        double sum = b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            sum += W1[j * HIDDEN_SIZE + i] * s_input[j];
        //hidden[i] = sum;
        
        //this is basically relu function
        hidden[i] = (sum > 0) ? sum : 0;
    }
}


__global__ void compute_output(double* W2, double* b2, double* hidden, double* output) {

    extern __shared__ double s_hidden[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Load hidden layer into shared memory
    if (tid < HIDDEN_SIZE) {
        s_hidden[tid] = hidden[tid];
    }
    __syncthreads();

    if (i < OUTPUT_SIZE) {
        double sum = b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            sum += W2[j * OUTPUT_SIZE + i] * s_hidden[j];
        output[i] = sum;
    }
    __syncthreads();
    if(i == 0){
        kernelsoftmax(output, OUTPUT_SIZE);
    }
}

//__global__ void apply_softmax(double* output) {
//    kernelsoftmax(output, OUTPUT_SIZE);
//}

// Backward Kernels
__global__ void compute_d_output(double* output, double* target, double* d_output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < OUTPUT_SIZE)
        d_output[i] = output[i] - target[i];
}

__global__ void compute_d_hidden(double* W2, double* d_output, double* hidden, double* d_hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN_SIZE) {
        double sum = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            sum += W2[i * OUTPUT_SIZE + j] * d_output[j];
        d_hidden[i] = sum * ((hidden[i] > 0) ? 1.0 : 0.0);
    }
}

__global__ void update_W2(double* W2, double* d_output, double* hidden, double lr) {

    int j = threadIdx.x;  // [0, HIDDEN_SIZE)
    int i = blockIdx.x;   // [0, OUTPUT_SIZE)
    
    if (j < HIDDEN_SIZE && i < OUTPUT_SIZE) {
        // CORRECT TRANSPOSED UPDATE
        W2[j * OUTPUT_SIZE + i] -= lr * d_output[i] * hidden[j];
    }
}

__global__ void update_W1(double* W1, double* d_hidden, double* input, double lr) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int i = idx / INPUT_SIZE;
    // int j = idx % INPUT_SIZE;
    // if (i < HIDDEN_SIZE && j < INPUT_SIZE)
    //     W1[i * INPUT_SIZE + j] -= lr * d_hidden[i] * input[j];

    int input_idx = blockIdx.x;    // [0, INPUT_SIZE)
    int hidden_idx = threadIdx.x;  // [0, HIDDEN_SIZE)
    
    if (input_idx < INPUT_SIZE && hidden_idx < HIDDEN_SIZE) {
        // CORRECT TRANSPOSED UPDATE
        int weight_idx = input_idx * HIDDEN_SIZE + hidden_idx;
        W1[weight_idx] -= lr * d_hidden[hidden_idx] * input[input_idx];
    }
}

__global__ void update_b1(double* b1, double* d_hidden, double lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN_SIZE)
        b1[i] -= lr * d_hidden[i];
}

__global__ void update_b2(double* b2, double* d_output, double lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < OUTPUT_SIZE)
        b2[i] -= lr * d_output[i];
}

// Data loading functions (same as before)
double* loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fseek(file, 16, SEEK_SET);
    double* images = allocateContiguous(numImages * INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i * INPUT_SIZE + j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

// ---------------------------------------
// Load MNIST Labels from file into a contiguous array.
// Returns a pointer to an array of doubles of length (numLabels * OUTPUT_SIZE).
// Each label is one-hot encoded.
double* loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fseek(file, 8, SEEK_SET);
    double* labels = allocateContiguous(numLabels * OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i * OUTPUT_SIZE + j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

void calculateOccupancyFORWARD(int threadsPerBlock, size_t sharedMemBytes, cudaFuncCache cacheConfig) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int numBlocks;
    int activeWarps;
    int maxWarps = prop.maxThreadsPerMultiProcessor / 32; // 48 for RTX 3080
    
    // Set function attributes
    cudaFuncSetCacheConfig(compute_hidden, cacheConfig);
    
    // Calculate occupancy
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, compute_hidden, threadsPerBlock, sharedMemBytes);
    activeWarps = numBlocks * (threadsPerBlock / 32);
    
    float occupancy = (float)activeWarps / maxWarps;
    
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Shared memory per block: %zu bytes\n", sharedMemBytes);
    printf("Blocks per SM: %d\n", numBlocks);
    printf("Active warps: %d\n", activeWarps);
    printf("Theoretical occupancy: %.2f%%\n\n", occupancy * 100);
}

void forwardKernelLaunching(double* d_W1, double* d_W2, double* d_b1, double* d_b2,double* d_input,double *D_hidden, double* D_output,
                            int sharedMemINPUT, int sharedMemHIDDEN){
    compute_hidden<<<4, 128, sharedMemINPUT>>>(d_W1, d_b1, d_input, D_hidden);
    compute_output<<<4, 128, sharedMemHIDDEN>>>(d_W2, d_b2, D_hidden, D_output);
}

void backwardKernelLaunching(double* d_W1, double* d_W2, double* d_b1, double* d_b2,double* d_input,double *D_hidden, double* D_output,
                             double* d_label, double *D_d_hidden, double *D_d_output, cudaStream_t compute_stream, cudaStream_t copy_stream, 
                             cudaEvent_t copy_event, double* h_output){

    //compute_d_output<<<1, OUTPUT_SIZE>>>(D_output, d_label, D_d_output);
    compute_d_output<<<1, OUTPUT_SIZE, 0, compute_stream>>>(D_output, d_label, D_d_output);

    //Record the event in the compute stream
    cudaEventRecord(copy_event, compute_stream);

    //In the copy stream, wait for the event and then asynchronously copy D_output to h_output.
    cudaStreamWaitEvent(copy_stream, copy_event, 0);
    cudaMemcpyAsync(h_output, D_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost, copy_stream);

    //In the compute stream, continue with the dependent backward kernels.
    //Since these are launched in the same stream as compute_d_output, their ordering is guaranteed.
    compute_d_hidden<<<4, 32, 0, compute_stream>>>(d_W2, D_d_output, D_hidden, D_d_hidden);
    update_W2<<<OUTPUT_SIZE, HIDDEN_SIZE, 0, compute_stream>>>(d_W2, D_d_output, D_hidden, LEARNING_RATE);
    update_W1<<<(HIDDEN_SIZE * INPUT_SIZE + 255) / 256, 256, 0, compute_stream>>>(d_W1, D_d_hidden, d_input, LEARNING_RATE);
    update_b1<<<4, 128, 0, compute_stream>>>(d_b1, D_d_hidden, LEARNING_RATE);
    update_b2<<<1, OUTPUT_SIZE, 0, compute_stream>>>(d_b2, D_d_output, LEARNING_RATE);
}

// Training function
void train(NeuralNetwork* net, double* d_W1, double* d_W2, double* d_b1, double* d_b2,
           double* H_images, double* H_labels, int numImages) {

    double *D_images, *D_labels;
    cudaMalloc(&D_images, numImages * INPUT_SIZE * sizeof(double));
    cudaMalloc(&D_labels, numImages * OUTPUT_SIZE * sizeof(double));
    cudaMemcpy(D_images, H_images, numImages * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(D_labels, H_labels, numImages * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    double *D_hidden, *D_output, *D_d_output, *D_d_hidden;
    cudaMalloc(&D_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&D_output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&D_d_output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&D_d_hidden, HIDDEN_SIZE * sizeof(double));

    dim3 block(256);
    dim3 grid_hidden((HIDDEN_SIZE + block.x - 1) / block.x);
    dim3 grid_output((OUTPUT_SIZE + block.x - 1) / block.x);

    cudaEvent_t f_start, f_stop, b_start, b_stop;
    cudaEventCreate(&f_start);
    cudaEventCreate(&f_stop);
    // bind backward events to compute_stream so they measure only GPU work,
    // not host synchronization
    cudaEventCreateWithFlags(&b_start, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&b_stop , cudaEventBlockingSync);
    float forward_ms = 0.0f, backward_ms = 0.0f;

    // - compute_stream: for all backward kernels (ensuring ordering)
    // - copy_stream: for asynchronous memory copy.
    cudaStream_t compute_stream, copy_stream;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&copy_stream);

    // Create an event to synchronize between the streams.
    cudaEvent_t copy_event;
    cudaEventCreate(&copy_event);

    // Allocate pinned host memory for the asynchronous copy.
    double* h_output;
    cudaMallocHost(&h_output, OUTPUT_SIZE * sizeof(double));  // Pinned host memory
    
    int sharedMemINPUT = INPUT_SIZE*sizeof(double);
    int sharedMemHidden = HIDDEN_SIZE*sizeof(double);

    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        forward_ms = backward_ms = 0.0f;

        for (int i = 0; i < numImages; i++) {
            double* d_input = D_images + i * INPUT_SIZE;
            double* d_label = D_labels + i * OUTPUT_SIZE;

            // Forward Pass
            cudaEventRecord(f_start, 0);
            forwardKernelLaunching(d_W1,d_W2,d_b1,d_b2,d_input,D_hidden,D_output, sharedMemINPUT, sharedMemHidden);
            cudaEventRecord(f_stop, 0);
            cudaDeviceSynchronize();
            {
                float ms;
                cudaEventElapsedTime(&ms, f_start, f_stop);
                forward_ms += ms;
            }

            cudaEventRecord(b_start, compute_stream);
            // Backward Pass
            backwardKernelLaunching(d_W1,d_W2,d_b1,d_b2,d_input,D_hidden,D_output, d_label, D_d_hidden, D_d_output,compute_stream, copy_stream, copy_event, h_output);

            // Synchronize both streams before computing loss/accuracy.
            cudaStreamSynchronize(copy_stream);
            cudaEventRecord(b_stop, compute_stream);
            cudaEventSynchronize(b_stop);            {
                float ms;
                cudaEventElapsedTime(&ms, b_start, b_stop);
                backward_ms += ms;
            }

            for (int k = 0; k < OUTPUT_SIZE; k++)
                loss -= H_labels[i * OUTPUT_SIZE + k] * log(h_output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (h_output[j] > h_output[pred]) pred = j;
                if (H_labels[i * OUTPUT_SIZE + j] > H_labels[i * OUTPUT_SIZE + actual]) actual = j;
            }
            if (pred == actual) correct++;
            
            cudaStreamSynchronize(compute_stream);

        }

        printf("Epoch %d - Loss: %.4f - Acc: %.2f%% - Time: %.3fs - (FWD: %.3fs, BWD: %.3fs)\n",
               epoch + 1, loss / numImages, (correct * 100.0) / numImages, get_time(epoch_start), forward_ms/1000, backward_ms/1000);
    }
    printf("Total Time: %.3fs\n", get_time(total_start));

    cudaFree(D_images);
    cudaFree(D_labels);
    cudaFree(D_hidden);
    cudaFree(D_output);
    cudaFree(D_d_output);
    cudaFree(D_d_hidden);
}

// Evaluation function
void evaluate(double* images, double* labels, int numImages, double* d_W1, double* d_W2, double* d_b1, double* d_b2) {
    double *D_hidden, *D_output;
    cudaMalloc(&D_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&D_output, OUTPUT_SIZE * sizeof(double));

    dim3 block(256);
    dim3 grid_hidden((HIDDEN_SIZE + block.x - 1) / block.x);
    dim3 grid_output((OUTPUT_SIZE + block.x - 1) / block.x);
    int sharedMemINPUT = INPUT_SIZE*sizeof(double);
    int sharedMemHidden = HIDDEN_SIZE*sizeof(double);
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double* d_input;
        cudaMalloc(&d_input, INPUT_SIZE * sizeof(double));
        cudaMemcpy(d_input, images + i * INPUT_SIZE, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

        forwardKernelLaunching(d_W1,d_W2,d_b1,d_b2,d_input,D_hidden,D_output, sharedMemINPUT,sharedMemHidden);
        cudaDeviceSynchronize();

        double h_output[OUTPUT_SIZE];
        cudaMemcpy(h_output, D_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > h_output[pred]) pred = j;
            if (labels[i * OUTPUT_SIZE + j] > labels[i * OUTPUT_SIZE + actual]) actual = j;
        }
        if (pred == actual) correct++;
        cudaFree(d_input);
    }

    printf("Test Acc: %.2f%%\n", (correct * 100.0) / numImages);
    cudaFree(D_hidden);
    cudaFree(D_output);
}

// Main function (same as before with added CUDA allocations)
int main() {
    double* H_train_images = loadMNISTImages("../../data/train-images.idx3-ubyte", 60000);
    double* H_train_labels = loadMNISTLabels("../../data/train-labels.idx1-ubyte", 60000);
    double* H_test_images = loadMNISTImages("../../data/t10k-images.idx3-ubyte", 10000);
    double* H_test_labels = loadMNISTLabels("../../data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();

    double *d_W1, *d_W2, *d_b1, *d_b2;
    cudaMalloc(&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc(&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_b1, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_b2, OUTPUT_SIZE * sizeof(double));

    cudaMemcpy(d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    train(net, d_W1, d_W2, d_b1, d_b2, H_train_images, H_train_labels, 60000);
    evaluate(H_test_images, H_test_labels, 10000, d_W1, d_W2, d_b1, d_b2);

    freeNetwork(net);
    free(H_train_images);
    free(H_train_labels);
    free(H_test_images);
    free(H_test_labels);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_b1);
    cudaFree(d_b2);

    return 0;
}